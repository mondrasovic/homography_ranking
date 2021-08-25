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

from typing import Iterable, List, Optional

import numpy as np

from datagen.position import FixedPositionGen, PositionGenerator
from datagen.shape import ShapeGenerator


def build_grid_layout(
        n_rows: int, n_cols: int, *, x_border: float = 0.2,
        y_border: float = 0.2
) -> List[PositionGenerator]:
    assert n_rows > 0 and n_cols > 0
    assert 0 < x_border < 0.5 and 0 < y_border < 0.5
    
    x_seps = np.linspace(x_border, 1 - x_border, n_cols)
    y_seps = np.linspace(y_border, 1 - y_border, n_rows)
    xs, ys = np.meshgrid(x_seps, y_seps)
    
    return [FixedPositionGen(x, y) for x, y in zip(xs.flatten(), ys.flatten())]


class ShapesLayoutGenerator:
    """Generates vertices of given shape anchored in a specific layout.
    """
    
    def __init__(
            self, img_width: int, img_height: int, shape_gen: ShapeGenerator,
            position_gens: Iterable[PositionGenerator],
            n_shapes: Optional[int] = None
    ) -> None:
        self.img_width: int = img_width
        self.img_height: int = img_height
        
        self.shape_gen: ShapeGenerator = shape_gen
        self.position_gens: Iterable[PositionGenerator] = position_gens
        self.n_shapes: Optional[int] = n_shapes
    
    @property
    def shape_name(self) -> str:
        return self.shape_gen.shape_name
    
    def generate_points(self) -> np.ndarray:
        points = map(
            self.shape_gen.generate_points,
            self._generate_center_points()
        )
        points = np.array(tuple(points), dtype=np.float)
        
        # Select only a random subset of the points.
        if self.n_shapes is not None:
            shapes_num = min(len(points), self.n_shapes)
            selector = np.array(
                [True] * shapes_num + [False] * (len(points) - shapes_num)
            )
            np.random.shuffle(selector)
            points = points[selector]
        
        return points
    
    def _generate_center_points(self) -> Iterable[np.ndarray]:
        return map(
            lambda g: g.generate_absolute_position(
                self.img_width, self.img_height),
            self.position_gens
        )
