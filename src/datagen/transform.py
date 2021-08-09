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
from typing import Callable, Iterable, Optional, Union

import numpy as np


CoordShiftT = Union[Callable[[np.ndarray], np.ndarray], np.ndarray]
ScaleFactorT = Union[Callable[[np.ndarray], float], float]
AngleT = Union[Callable[[np.ndarray], float], float]


class Transformation(abc.ABC):
    """
    A general transformation that is applied for an array of points
    (pixel coordinates).
    """
    
    @abc.abstractmethod
    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms an array of points (pixel coordinates).
        
        :param points: points to transform
        """
        pass


class GroupTransformation(Transformation, abc.ABC):
    """
    A general transformation that allows to transform entire groups of
    points, i.e. a 3D tensor, not just a single 2D array.
    """
    
    def transform(self, points: np.ndarray) -> np.ndarray:
        # docstring inherited
        assert 2 <= points.ndim <= 3
        
        if points.ndim == 2:
            return self.transform_points_group(points)
        else:
            return np.stack([self.transform_points_group(p) for p in points])
    
    @abc.abstractmethod
    def transform_points_group(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms a single 2D groups of points.

        :param points: points to transform
        """
        pass


class Identity(Transformation):
    """
    Identity transformation. Causes no modification.
    """
    
    def transform(self, points: np.ndarray) -> np.ndarray:
        # docstring inherited
        return points


class Translation(GroupTransformation):
    """
    Translation transformation. Translates (shifts) the points in a
    specific way. A custom shift generator may be supplied (for instance,
    to obtain a random shift according to some probability distribution),
    or a fixed point.
    """
    
    def __init__(self, coord_shift: CoordShiftT) -> None:
        """
        Constructor.

        :param coord_shift: specifies the translation either as a fixed point or
        a callable that takes the current points as an input and returns a new
        points
        """
        self.coord_shift: CoordShiftT = coord_shift
    
    def transform_points_group(self, points: np.ndarray) -> np.ndarray:
        # docstring inherited
        shift = self.coord_shift
        if callable(self.coord_shift):
            shift = self.coord_shift(points)
        return points + shift


class Scale(GroupTransformation):
    """
    Scaling transformation. Multiplies points by a specific factor. It
    allows the scaling origin to be adjusted (for example, if points should be
    scaled with respect to their centroid).
    """

    def __init__(
            self, factor: ScaleFactorT = 1.0,
            origin: Optional[np.ndarray] = None, *,
            use_centroid: bool = False) -> None:
        """
        Constructor.

        :param factor: a positive factor to scale the points by, or a callable
        that returns a scale factor for the current batch of points
        :param origin: point (origin) with respect to which to scale the points
        (optional). If not set together with the flag to use the centroid, then
        the [0,0] point is assumed as the origin.
        :param use_centroid: use computed centroid from the given points as the
        origin (overrides the origin parameter)
        """
        self.factor: ScaleFactorT = factor
        self.origin: Optional[np.ndarray] = origin
        if not use_centroid and origin is None:
            self.origin = np.zeros(2)
        self.use_centroid: bool = use_centroid

    def transform_points_group(self, points: np.ndarray) -> np.ndarray:
        # docstring inherited
        origin = np.mean(points, axis=0) if self.use_centroid else self.origin
        factor = self.factor
        if callable(self.factor):
            factor = self.factor(points)
        assert factor > 0
        return (points - origin) * factor + origin


class Rotation(GroupTransformation):
    """
    Rotate point(s) counterclockwise by a given angle around a given origin.
    The angle is assumed to be in radians.
    """
    
    def __init__(
            self, angle: AngleT = 0.0, origin: Optional[np.ndarray] = None, *,
            use_centroid: bool = False) -> None:
        """
        Constructor.

        :param angle: angle in radians by which to rotate the points or a
        callable that returns an angle for the current batch of points
        :param origin: point (origin) around which to rotate the points
        (optional)
        :param use_centroid: use computed centroid from the given points as the
        origin (overrides the origin parameter)
        """
        assert (isinstance(origin, np.ndarray) or
                (origin is None and use_centroid))
        assert origin is None or (origin.ndim == 1 and len(origin) == 2)
        
        self.angle: AngleT = angle
        self.origin: Optional[np.ndarray] = origin
        self.use_centroid: bool = use_centroid
    
    def transform_points_group(self, points: np.ndarray) -> np.ndarray:
        # docstring inherited
        angle = self.angle
        if callable(self.angle):
            angle = self.angle(points)
        origin = np.mean(points, axis=0) if self.use_centroid else self.origin
        return self.rotate_points(points, angle, origin)
    
    @staticmethod
    def rotate_points(
            points: np.ndarray, angle: float, origin: np.ndarray) -> np.ndarray:
        """
        Rotate point(s) counterclockwise by a given angle around a given
        origin. The angle is assumed to be in radians.

        :param points: point(s) to rotate
        :param angle: angle in radians by which to rotate the point
        :param origin: point (origin) around which to rotate the point
        :return: rotated point(s) by a specific angle around a specific origin
        """
        sin_val, cos_val = np.sin(angle), np.cos(angle)
        
        rotation_matrix = np.array(((cos_val, -sin_val),
                                    (sin_val, cos_val)))
        origin = np.atleast_2d(origin)
        points = np.atleast_2d(points)
        
        return np.squeeze(
            (rotation_matrix @ (points.T - origin.T) + origin.T).T)


class Pipeline(Transformation):
    """
    A pipeline composed of multiple transformations.
    """
    
    def __init__(self, *args: Transformation) -> None:
        """
        Constructor.

        :param args: individual transformations to compose the pipeline from
        (order of execution is preserved)
        """
        self.transforms: Iterable[Transformation] = args
    
    def transform(self, points: np.ndarray) -> np.ndarray:
        # docstring inherited
        curr_points = points
        for transform in self.transforms:
            curr_points = transform.transform(curr_points)
        return curr_points
