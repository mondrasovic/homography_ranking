#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>
#
# MIT License
#
# Copyright (c) 2021 Milan Ondrašovič
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import dataclasses
import itertools
import math
import pathlib
import time
import multiprocessing as mp
import pickle
from typing import cast, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tqdm

from common.dtypes import ShapeT
from datagen.layout import ShapesLayoutGenerator
from datagen.position import PositionGenerator
from datagen.rotation import Rotation3DGenerator
from datagen.shape import ShapeGenerator
from datagen.transform import Pipeline, Transformation
from transform.base import HomographyTransformer
from transform.homographyranking import rank_homographies
from transform.perspective import PerspectiveTransformer


@dataclasses.dataclass(frozen=True)
class CommonTestSpec:
    img_shape_orig: ShapeT
    pix_coords_orig: np.ndarray
    img_3d_rotation_gen: Rotation3DGenerator
    positions_gens: List[PositionGenerator]
    rotation_transform: Transformation
    translation_transform: Transformation
    scale_transform: Transformation
    noise_transform: Transformation


@dataclasses.dataclass(frozen=True)
class TestScenarioSpec:
    id: int
    common: CommonTestSpec
    shape_gen: ShapeGenerator
    n_groups: int
    use_translation: bool = False
    use_rotation: bool = False
    use_scale: bool = False
    use_noise: bool = False


@dataclasses.dataclass(frozen=True)
class ErrorStats:
    min: float
    max: float
    mean: float
    median: float
    stdev: float


def _build_points_transforms(
        scneario: TestScenarioSpec
) -> Tuple[Optional[Transformation], Optional[Transformation]]:
    orig_transforms = []
    
    if scneario.use_translation:
        orig_transforms.append(scneario.common.translation_transform)
    if scneario.use_rotation:
        orig_transforms.append(scneario.common.rotation_transform)
    if scneario.use_scale:
        orig_transforms.append(scneario.common.scale_transform)
    
    if orig_transforms:
        if len(orig_transforms) == 1:
            orig_transforms = orig_transforms[0]
        else:
            orig_transforms = Pipeline(*orig_transforms)
    else:
        orig_transforms = None
    
    warped_transforms = None
    if scneario.use_noise:
        warped_transforms = scneario.common.noise_transform
    
    return orig_transforms, warped_transforms


def _eval_rectification(
        transformer: HomographyTransformer, pix_coords_orig: np.ndarray,
        pix_coords_warped: np.ndarray, img_shape_orig: ShapeT
) -> np.ndarray:
    pix_coords_rectified = transformer.transform_points(
        pix_coords_warped, img_shape_orig
    )
    return np.linalg.norm(pix_coords_orig - pix_coords_rectified, axis=1)


def run_experiment(
        scenario: TestScenarioSpec
) -> Tuple[TestScenarioSpec, Sequence[int], Sequence[ErrorStats], float]:
    common = scenario.common
    layout_gen = ShapesLayoutGenerator(
        img_width=common.img_shape_orig[1],
        img_height=common.img_shape_orig[0],
        shape_gen=scenario.shape_gen,
        position_gens=common.positions_gens,
        n_shapes=scenario.n_groups
    )
    points_orig_transform, points_warped_transform = _build_points_transforms(
        scenario
    )
    
    points_groups_orig = layout_gen.generate_points()
    
    if points_orig_transform:
        points_groups_orig = points_orig_transform.transform(
            points_groups_orig
        )
    rotation_transformer = common.img_3d_rotation_gen.build_transformer()
    
    # Reshape all the points belonging to individual shapes into a
    # list of 2D coordinates and then after perspective
    # transformation convert them back so that the structure is
    # preserved.
    points_groups_warped = rotation_transformer.transform_points(
        points_groups_orig.reshape(-1, 2),
        common.img_shape_orig, True
    ).reshape(points_groups_orig.shape)
    pix_coords_warped = rotation_transformer.transform_points(
        common.pix_coords_orig, common.img_shape_orig, True
    )
    
    if points_warped_transform is not None:
        points_groups_warped = points_warped_transform.transform(
            points_groups_warped
        )
    
    homographies, error_stats = [], []
    for points_orig, points_warped in zip(
            points_groups_orig, points_groups_warped
    ):
        transformer = PerspectiveTransformer.build_from_correspondences(
            points_warped, points_orig
        )
        homographies.append(transformer.homography)
        errors = _eval_rectification(
            transformer, common.pix_coords_orig, pix_coords_warped,
            common.img_shape_orig
        )
        curr_stats = ErrorStats(
            np.min(errors), np.max(errors), cast(float, np.mean(errors)),
            cast(float, np.median(errors)), cast(float, np.std(errors))
        )
        error_stats.append(curr_stats)
    
    start_time = time.process_time_ns()
    homography_order = rank_homographies(
        points_groups_warped, points_groups_orig, np.asarray(homographies)
    )
    time_elapsed_ns = time.process_time_ns() - start_time
    
    return scenario, homography_order, error_stats, time_elapsed_ns


def run_experiments(
        scenarios: Sequence[TestScenarioSpec], results_file_path: str,
        n_instances: int = 100
) -> None:
    assert results_file_path
    assert n_instances >= 0
    
    n_processes = max(1, mp.cpu_count() - 2)
    total_instances = len(scenarios) * n_instances
    chunk_size = int(math.ceil(total_instances / n_processes))
    
    experiment_spec_data = []
    single_homography_data = []
    homography_order_data = []
    
    def iter_params() -> Iterable[TestScenarioSpec]:
        for scenario in scenarios:
            yield from itertools.repeat(scenario, n_instances)
    
    inst_id = 1
    
    with mp.Pool(n_processes) as pool:
        with tqdm.tqdm(total=total_instances) as pbar:
            results = pool.imap_unordered(
                run_experiment, iter_params(), chunksize=chunk_size
            )
            
            for result in results:
                scenario, homography_order, error_stats, time_elapsed_ns =\
                    result
                experiment_spec_data.append({
                    'scenar_id': scenario.id,
                    'inst_id': inst_id,
                    'shape': scenario.shape_gen.shape_name,
                    'n_groups': scenario.n_groups,
                    'transl': scenario.use_translation,
                    'rot': scenario.use_rotation,
                    'scale': scenario.use_scale,
                    'noise': scenario.use_noise,
                    'time_elapsed_ns': time_elapsed_ns,
                })
                
                for i, curr_stats in enumerate(error_stats):
                    single_homography_data.append({
                        'inst_id': inst_id,
                        'group_idx': i,
                        'min': curr_stats.min,
                        'max': curr_stats.max,
                        'mean': curr_stats.mean,
                        'med': curr_stats.median,
                        'std': curr_stats.stdev,
                    })
                
                homography_order_data.append({
                    'inst_id': inst_id,
                    'order': homography_order,
                })
                
                inst_id += 1
                pbar.update()
    
    experiment_spec_df = pd.DataFrame(experiment_spec_data)
    single_homography_df = pd.DataFrame(single_homography_data)
    homography_order_df = pd.DataFrame(homography_order_data)
    
    pathlib.Path(results_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file_path, 'wb') as out_file:
        pickle.dump(
            (experiment_spec_df, single_homography_df, homography_order_df),
            out_file, protocol=pickle.HIGHEST_PROTOCOL
        )
