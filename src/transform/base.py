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
from typing import *

import cv2 as cv
import numpy as np

from common.dtypes import ShapeT


class ImageTransformer(abc.ABC):
    """An abstract image transformer that allows to transform images or just
    points.
    """
    
    @abc.abstractmethod
    def transform_image(
            self, img: np.ndarray, output_shape: Optional[ShapeT] = None,
            adaptive_output_shape: bool = False
    ) -> np.ndarray:
        """Transforms an image using the transformation properties with which
        an instance was built.

        :param img: a NumPy image to be transformed
        :param output_shape: (optional) a fixed output shape (if some dimension
        is smaller than the one after transformation, the image will be cropped
        in that dimension)
        :param adaptive_output_shape: automatically determine the output shape
        of the image with respect ot the transformation (overrides the
        output_shape parameter if both are provided)
        :return: a transformed image
        """
        pass
    
    @abc.abstractmethod
    def transform_points(
            self, points: np.ndarray, img_shape: ShapeT,
            adaptive_output_shape: bool = False
    ) -> np.ndarray:
        """Transforms points using the transformation properties with which an
        instance was built.

        :param points: one or more points to be transformed
        :param img_shape: image shape with respect to which the transformation
        is performed (beware that resulting points may still lie outside this
        range)
        :param adaptive_output_shape: automatically determine the output shape
        of the image with respect ot the transformation (overrides the
        output_shape parameter if both are provided)
        :return: a sequence of transformed points
        """
        pass


class HomographyTransformer(ImageTransformer, abc.ABC):
    
    @property
    @abc.abstractmethod
    def homography(self) -> np.ndarray:
        """Return the corresponding homography matrix performing the
        transformation.

        :return: the homography matrix
        """
        pass
    
    def transform_image(
            self, img: np.ndarray, output_shape: Optional[ShapeT] = None,
            adaptive_output_shape: bool = False) -> np.ndarray:
        # docstring inherited
        new_output_shape = output_shape
        if output_shape is None:
            new_output_shape = img.shape
        
        homography = self._build_homography(new_output_shape)
        
        if adaptive_output_shape:
            homography, new_output_shape = self._adapt_homography_for_shape(
                homography, img.shape)
        
        return cv.warpPerspective(
            img, homography, (new_output_shape[1], new_output_shape[0]))
    
    def transform_points(
            self, points: np.ndarray, img_shape: ShapeT,
            adaptive_output_shape: bool = False) -> np.ndarray:
        # docstring inherited
        homography = self._build_homography(img_shape)
        
        if adaptive_output_shape:
            homography, _ = self._adapt_homography_for_shape(
                homography, img_shape)
        
        return self._transform_points(points, homography)
    
    @abc.abstractmethod
    def _build_homography(
            self, img_shape: Optional[ShapeT] = None
    ) -> np.ndarray:
        """Builds a homography matrix.

        :param img_shape: image shape with respect to which the transformation
        should take place
        :return: a homography matrix
        """
        pass
    
    @staticmethod
    def _adapt_homography_for_shape(
            homography: np.ndarray,
            img_shape: ShapeT
    ) -> Tuple[np.ndarray, ShapeT]:
        corners = np.float32((
            (0, 0),
            (img_shape[1] - 1, 0),
            (img_shape[1] - 1, img_shape[0] - 1),
            (0, img_shape[0] - 1)
        ))
        corners_transformed = HomographyTransformer._transform_points(
            corners, homography
        )
        
        min_vals = np.min(corners_transformed, axis=0)
        max_vals = np.max(corners_transformed, axis=0)
        
        x_min, y_min = min_vals[0], min_vals[1]
        x_max, y_max = max_vals[0], max_vals[1]
        
        translation_mat = np.array((
            (1, 0, -x_min),
            (0, 1, -y_min),
            (0, 0, 1)
        ))
        homography = translation_mat.dot(homography)
        
        new_width, new_height = x_max - x_min, y_max - y_min
        new_output_shape = (int(new_height), int(new_width))
        
        return homography, new_output_shape
    
    @staticmethod
    def _transform_points(
            points: np.ndarray, homography: np.ndarray
    ) -> np.ndarray:
        return np.squeeze(
            cv.perspectiveTransform(points.reshape(-1, 1, 2), homography), 1
        )
