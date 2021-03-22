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

import cv2 as cv
import numpy as np

from common.dtypes import ShapeT
from transform.base import HomographyTransformer


class PerspectiveTransformer(HomographyTransformer):
    """
    A perspective transformer that applies a homography transformation.
    """
    
    def __init__(self, homography: np.ndarray) -> None:
        """
        Constructor.

        :param homography: a homography matrix performing the perspective
        transformation
        """
        self._homography = homography
    
    @property
    def homography(self) -> np.ndarray:
        # docstring inherited
        return self._homography
    
    @staticmethod
    def build_from_correspondences(
            src_points: np.ndarray,
            dst_points: np.ndarray) -> 'PerspectiveTransformer':
        """
        Builds a homography transformation based upon multiple (at least 4)
        point correspondences between two planes.

        :param src_points: points in the source plane
        :param dst_points: points in the destination (target) plane
        :return: an instance of the :class:`PerspectiveTransformer`
        """
        return PerspectiveTransformer(
            cv.findHomography(src_points, dst_points, method=cv.RANSAC)[0])
    
    def _build_homography(self, img_shape: ShapeT = None) -> np.ndarray:
        return self._homography
