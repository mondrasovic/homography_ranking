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

import abc
import itertools

import numpy as np


class ShapeGenerator(abc.ABC):
    def __init__(self, shape_name: str) -> None:
        self.shape_name: str = shape_name
    
    @abc.abstractmethod
    def generate_points(self, shape_center: np.ndarray) -> np.ndarray:
        pass


class RectangleGenerator(ShapeGenerator):
    def __init__(self, width: float, height: float) -> None:
        super().__init__('rectangle')
        
        self.width: float = width
        self.height: float = height
    
    def generate_points(self, shape_center: np.ndarray) -> np.ndarray:
        width_half, height_half = self.width / 2, self.height / 2
        
        top_left = (shape_center[0] - width_half, shape_center[1] - height_half)
        top_right = (shape_center[0] + width_half, top_left[1])
        bottom_left = (top_left[0], shape_center[1] + height_half)
        bottom_right = (top_right[0], bottom_left[1])
        
        return np.array(
            (top_left, top_right, bottom_right, bottom_left), dtype=np.float
        )


class PolygonGenerator(ShapeGenerator):
    def __init__(self, radius: float, n_vertices: int) -> None:
        assert n_vertices >= 4
        
        super().__init__(f'{n_vertices}-polygon')
        
        self.radius: float = radius
        self.n_vertices: int = n_vertices
    
    def generate_points(self, shape_center: np.ndarray) -> np.ndarray:
        step = np.pi * 2 / self.n_vertices
        angles = np.fromiter(
            itertools.count(0, step), np.float, self.n_vertices
        )
        xs, ys = np.cos(angles), np.sin(angles)
        points = np.stack((xs, ys)) * self.radius + shape_center[..., None]
        return points.T


if __name__ == '__main__':
    import functools
    
    import cv2 as cv
    
    win_name = "Shape generating"
    
    img_width, img_height, image_depth = 1000, 1000, 3
    img_shape = (img_height, img_width, image_depth)
    shape_center = np.array((img_width // 2, img_height // 2))
    radius = min(img_width, img_height) // 4
    n_vertices = 5
    
    
    def update_image():
        image = np.zeros(img_shape)
        polygon_gen = PolygonGenerator(radius, n_vertices)
        points = polygon_gen.generate_points(shape_center)
        points = points.round().astype(np.int)
        cv.polylines(
            image, [points], True, color=(0, 255, 0), thickness=3,
            lineType=cv.LINE_AA
        )
        cv.imshow(win_name, image)
    
    
    def on_shape_center_trackbar_change(index, value):
        global shape_center
        shape_center[index] = value
        update_image()
    
    
    def on_radius_trackbar_change(value):
        global radius
        radius = value
        update_image()
    
    
    cv.namedWindow(win_name)
    cv.createTrackbar(
        "center X", win_name, shape_center[0], img_width,
        functools.partial(on_shape_center_trackbar_change, 0)
    )
    cv.createTrackbar(
        "center Y", win_name, shape_center[1], img_height,
        functools.partial(on_shape_center_trackbar_change, 1)
    )
    cv.createTrackbar(
        "radius", win_name, radius, img_width, on_radius_trackbar_change
    )
    
    update_image()
    
    cv.waitKey(0)
    cv.destroyAllWindows()
