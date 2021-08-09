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

import sys
import numpy as np

import config
from common.dtypes import ShapeT
from common.utils import rand_uniform
from datagen.layout import build_grid_layout
from datagen.rotation import Rotation3DGenerator
from datagen.shape import PolygonGenerator, RectangleGenerator
from datagen.transform import Rotation, Scale, Translation
from experiment.runner import CommonTestSpec, run_experiments, TestScenarioSpec


def get_img_pix_coords_matrix(img_shape: ShapeT) -> np.ndarray:
    """Create a matrix with all the [x, y] coordinates of individual pixels
    for an image of a specific shape. Coordinates are 0-indexed.

    :param img_shape: image shape
    :return: a matrix containing [x, y] 0-based coordinates of all the
    pixels
    """
    cols, rows = np.meshgrid(
        np.arange(img_shape[1]), np.arange(img_shape[0])
    )
    return np.concatenate(
        (cols[..., None], rows[..., None]), axis=2
    ).reshape(-1, 2)


def rand_rotate_angles(points: np.ndarray) -> float:
    return rand_uniform(0, 2 * np.pi)


def rand_translation_coefs(points: np.ndarray) -> np.ndarray:
    return rand_uniform(-20, 20, points.shape[-1])


def rand_scale_coef(points: np.ndarray) -> float:
    return rand_uniform(0.8, 1.5)


def rand_noise_coefs(points: np.ndarray) -> np.ndarray:
    return rand_uniform(-2, 2, points.shape)


def main() -> int:
    np.random.seed(731995)
    
    img_shape_orig = (config.IMG_HEIGHT, config.IMG_WIDTH)
    img = np.zeros(img_shape_orig)
    pix_coords_orig = get_img_pix_coords_matrix(img.shape).astype(np.float)
    
    rotation_range = config.IMG_ROTATION_RANGE
    rotation_gen = Rotation3DGenerator(
        x_axis_range=rotation_range, y_axis_range=rotation_range,
        z_axis_range=rotation_range
    )
    
    positions_gens = build_grid_layout(3, 3)
    
    rand_rotate = Rotation(rand_rotate_angles, use_centroid=True)
    rand_translate = Translation(rand_translation_coefs)
    # Using non-uniform scale.
    rand_scale = Scale(rand_scale_coef, use_centroid=True)
    rand_noise = Translation(rand_noise_coefs)
    
    common_spec = CommonTestSpec(
        img_shape_orig, pix_coords_orig, rotation_gen, positions_gens,
        rand_rotate, rand_translate, rand_scale, rand_noise
    )
    
    rectangle_gen = RectangleGenerator(
        config.SHAPE_BOX_WIDTH, config.SHAPE_BOX_HEIGHT
    )
    radius = min(config.SHAPE_BOX_WIDTH, config.SHAPE_BOX_HEIGHT) / 2
    polygon_5_gen = PolygonGenerator(radius, 5)
    polygon_7_gen = PolygonGenerator(radius, 7)
    polygon_9_gen = PolygonGenerator(radius, 9)
    
    scenario_id = 1
    
    def _scenario(**kwargs) -> TestScenarioSpec:
        nonlocal scenario_id
        scenario = TestScenarioSpec(
            id=scenario_id, common=common_spec, **kwargs
        )
        scenario_id += 1
        return scenario
    
    std_shape = dict(shape_gen=rectangle_gen, n_groups=6)
    all_affine = dict(use_translation=True, use_rotation=True, use_scale=True)
    
    scenarios = (
        _scenario(**std_shape),
        
        _scenario(use_translation=True, **std_shape),
        _scenario(use_rotation=True, **std_shape),
        _scenario(use_scale=True, **std_shape),
        
        _scenario(use_noise=False, **std_shape, **all_affine),
        _scenario(use_noise=True, **std_shape, **all_affine),
        
        _scenario(
            use_noise=True, shape_gen=polygon_5_gen, n_groups=6, **all_affine
        ),
        _scenario(
            use_noise=True, shape_gen=polygon_7_gen, n_groups=6, **all_affine
        ),
        _scenario(
            use_noise=True, shape_gen=polygon_9_gen, n_groups=6, **all_affine
        ),

        _scenario(
            use_noise=True, shape_gen=rectangle_gen, n_groups=3, **all_affine
        ),
        _scenario(
            use_noise=True, shape_gen=rectangle_gen, n_groups=5, **all_affine
        ),
        _scenario(
            use_noise=True, shape_gen=rectangle_gen, n_groups=7, **all_affine
        ),
        _scenario(
            use_noise=True, shape_gen=rectangle_gen, n_groups=9, **all_affine
        ),
    )
    
    run_experiments(
        scenarios, config.RESULTS_FILE_PATH, config.N_INSTANCES_PER_SCENARIO
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
