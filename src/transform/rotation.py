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

from typing import *

import numpy as np

from transform.base import HomographyTransformer, ShapeT


class Rotation3DTransformer(HomographyTransformer):
    """
    Transforms (rotation and/or translation) entire images or just points in
    3D space. The instance is meant to be immutable because of performance
    demands.
    """
    
    def __init__(
            self, theta: float = 0, phi: float = 0, gamma: float = 0,
            dx: float = 0, dy: float = 0, dz: float = 0):
        """
        Constructors the transformer.

        :param theta: x-axis rotation angle (in radians)
        :param phi: y-axis rotation angle (in radians)
        :param gamma: z-axis rotation angle (in radians)
        :param dx: translation in x-axis
        :param dy: translation in y-axis
        :param dz: translation in z-axis
        """
        self._theta: float = theta
        self._phi: float = phi
        self._gamma: float = gamma
        
        self._dx: float = dx
        self._dy: float = dy
        self._dz: float = dz
        
        self._rotation_mat = self.rotation_matrix(theta, phi, gamma)
    
    @property
    def homography(self) -> np.ndarray:
        # docstring inherited
        return self._rotation_mat
    
    @property
    def theta(self) -> float:
        """
        Returns the x-axis rotation angle (in radians).
        """
        return self._theta
    
    @property
    def phi(self) -> float:
        """
        Returns the y-axis rotation angle (in radians).
        """
        return self._phi
    
    @property
    def gamma(self) -> float:
        """
        Returns the z-axis rotation angle (in radians).
        """
        return self._gamma
    
    @property
    def dx(self) -> float:
        """
        Returns the translation in the x-axis.
        """
        return self._dx
    
    @property
    def dy(self) -> float:
        """
        Returns the translation in the y-axis.
        """
        return self._dy
    
    @property
    def dz(self) -> float:
        """
        Returns the translation in the z-axis.
        """
        return self._dz
    
    def _build_homography(
            self, img_shape: Optional[ShapeT] = None) -> np.ndarray:
        return self._build_homography_for_shape(img_shape)
    
    @staticmethod
    def projection_2d3d_matrix(img_shape: ShapeT) -> np.ndarray:
        return np.float32(((1, 0, -(img_shape[1] / 2.0)),
                           (0, 1, -(img_shape[0] / 2.0)),
                           (0, 0, 1),
                           (0, 0, 1)))
    
    @staticmethod
    def projection_3d2d_matrix(
            img_shape: ShapeT, focal: float) -> np.ndarray:
        return np.float32(((focal, 0, img_shape[1] / 2.0, 0),
                           (0, focal, img_shape[0] / 2.0, 0),
                           (0, 0, 1, 0)))
    
    @staticmethod
    def rotation_matrix(
            theta: float = 0, phi: float = 0, gamma: float = 0) -> np.ndarray:
        x_rotation_mat = Rotation3DTransformer.x_axis_rotation_matrix(theta)
        y_rotation_mat = Rotation3DTransformer.y_axis_rotation_matrix(phi)
        z_rotation_mat = Rotation3DTransformer.z_axis_rotation_matrix(gamma)
        
        return np.dot(np.dot(x_rotation_mat, y_rotation_mat), z_rotation_mat)
    
    @staticmethod
    def x_axis_rotation_matrix(theta: float = 0) -> np.ndarray:
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        
        return np.float32(((1, 0, 0, 0),
                           (0, cos_theta, -sin_theta, 0),
                           (0, sin_theta, cos_theta, 0),
                           (0, 0, 0, 1)))
    
    @staticmethod
    def y_axis_rotation_matrix(phi: float = 0) -> np.ndarray:
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        
        return np.float32(((cos_phi, 0, -sin_phi, 0),
                           (0, 1, 0, 0),
                           (sin_phi, 0, cos_phi, 0),
                           (0, 0, 0, 1)))
    
    @staticmethod
    def z_axis_rotation_matrix(gamma: float = 0) -> np.ndarray:
        sin_gamma, cos_gamma = np.sin(gamma), np.cos(gamma)
        
        return np.float32(((cos_gamma, -sin_gamma, 0, 0),
                           (sin_gamma, cos_gamma, 0, 0),
                           (0, 0, 1, 0),
                           (0, 0, 0, 1)))
    
    @staticmethod
    def translation_matrix(
            dx: float = 0, dy: float = 0, dz: float = 0) -> np.ndarray:
        return np.float32(((1, 0, 0, dx),
                           (0, 1, 0, dy),
                           (0, 0, 1, dz),
                           (0, 0, 0, 1)))
    
    def _build_homography_for_shape(self, output_shape: ShapeT) -> np.ndarray:
        focal = self._calc_focal_length(output_shape)
        
        projection_2d3d_mat = self.projection_2d3d_matrix(output_shape)
        projection_3d2d_mat = self.projection_3d2d_matrix(output_shape, focal)
        translation_mat = self.translation_matrix(self.dx, self.dy, focal)
        
        return np.dot(
            projection_3d2d_mat,
            np.dot(translation_mat,
                   np.dot(self._rotation_mat, projection_2d3d_mat)))
    
    def _calc_focal_length(self, output_shape: ShapeT) -> float:
        sin_gamma = np.sin(self._gamma)
        dist = np.linalg.norm(output_shape)
        focal = dist / (2 * sin_gamma if sin_gamma != 0 else 1)
        
        return focal
