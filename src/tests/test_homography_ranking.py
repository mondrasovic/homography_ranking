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

import unittest

import numpy as np

from transform.homographyranking import (
    _convert_to_homogeneous, _transform_points, _eval_reprojection_error,
    _transform_points_groups_stacked
)


class TestHomographyRanking(unittest.TestCase):
    def setUp(self) -> None:
        self.identity_homography = np.float32([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
    
    def test_convert_to_homogeneous_single_groups(self):
        points = np.asarray([
            [2, 1],
            [3, 5],
            [6, 2],
        ])
        points_homo = _convert_to_homogeneous(points)
        self.assertEqual(points_homo.tolist(), [
            [2, 3, 6],
            [1, 5, 2],
            [1, 1, 1],
        ])
    
    def test_convert_to_homogeneous_multiple_groups(self):
        points = np.asarray([
            [
                [2, 1],
                [3, 5],
                [6, 2],
            ],
            [
                [9, 4],
                [-7, 6],
                [2, -5],
            ]
        ])
        points_homo = _convert_to_homogeneous(points)
        self.assertEqual(points_homo.tolist(), [
            [
                [2, 3, 6],
                [1, 5, 2],
                [1, 1, 1],
            ],
            [
                [9, -7, 2],
                [4, 6, -5],
                [1, 1, 1],
            ],
        ])
    
    def test_convert_to_homogeneous_invalid_shape_1_dim(self):
        with self.assertRaises(ValueError):
            points = np.ones((2,))
            _convert_to_homogeneous(points)
    
    def test_convert_to_homogeneous_invalid_shape_4_dims(self):
        with self.assertRaises(ValueError):
            points = np.ones((2, 4, 6, 1))
            _convert_to_homogeneous(points)
    
    def test_transform_points_identity(self):
        points = np.float32([
            [1, 2],
            [3, 4],
            [5, 6],
        ])
        points_transformed = _transform_points(points, self.identity_homography)
        self.assertEqual(points_transformed.tolist(), [
            [1, 2],
            [3, 4],
            [5, 6],
        ])
    
    def test_transform_points_translation(self):
        x_shift, y_shift = 10, -10
        points = np.float32([
            [1, 2],
            [3, 4],
        ])
        homography = np.float32([
            [1, 0, x_shift],
            [0, 1, y_shift],
            [0, 0, 1],
        ])
        points_transformed = _transform_points(points, homography)
        self.assertEqual(points_transformed.tolist(), [
            [1 + x_shift, 2 + y_shift],
            [3 + x_shift, 4 + y_shift],
        ])
    
    def test_transform_points_1_dim(self):
        with self.assertRaises(ValueError):
            points = np.ones((2,), dtype=np.float32)
            _transform_points(points, self.identity_homography)

    def test_transform_points_3_dims(self):
        with self.assertRaises(ValueError):
            points = np.ones((2, 4, 2), dtype=np.float32)
            _transform_points(points, self.identity_homography)
    
    def test_transform_points_3D_points(self):
        with self.assertRaises(ValueError):
            points = np.ones((4, 3), dtype=np.float32)
            _transform_points(points, self.identity_homography)
    
    def test_eval_reprojection_error_equal_points(self):
        rectified_points = np.ones((3, 2))
        target_pooints = np.ones((3, 2))
        error = _eval_reprojection_error(rectified_points, target_pooints)
        self.assertEqual(error, 0)
    
    def test_eval_reprojection_error_diff_points(self):
        rectified_points = np.asarray([
            [1, 1],
            [2, 2],
        ])
        target_pooints = np.asarray([
            [2, 2],
            [3, 3],
        ])
        
        error = _eval_reprojection_error(rectified_points, target_pooints)
        self.assertEqual(error, 2)
    
    def test_transform_points_groups_stacked_affine_translate(self):
        points_groups = np.asarray([
            [
                [1, 2, 3],
                [2, 3, 4],
                [1, 1, 1],
            ],
            [
                [3, 4, 5],
                [6, 7, 8],
                [1, 1, 1],
            ],
        ])
        affine_matrices = np.asarray([
            [
                [1, 0, 1],  # Translate X and Y by 1.
                [0, 1, 1],
                [0, 0, 1],
            ],
            [
                [1, 0, 2],  # Translate X and Y by 2.
                [0, 1, 2],
                [0, 0, 1],
            ],
        ])
        
        points_transformed = _transform_points_groups_stacked(
            self.identity_homography, affine_matrices, points_groups)
        
        self.assertEqual(points_transformed.tolist(), [
            [
                [2, 3, 4],
                [3, 4, 5],
                [1, 1, 1],
            ],
            [
                [5, 6, 7],
                [8, 9, 10],
                [1, 1, 1],
            ],
        ])


if __name__ == '__main__':
    unittest.main()
