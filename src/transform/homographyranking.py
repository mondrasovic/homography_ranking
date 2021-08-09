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

from typing import cast, Sequence

import cv2 as cv
import numpy as np


class TransformationNotFoundError(Exception):
    """Raised when no transformation could be found for the given point
    correspondence.
    """
    pass


def _convert_to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Converts points to homogeneous coordinates by reshaping the array and
    adding "1" in the z dimension.

    :param points: array of matrices of shape either GxKx2 or Kx2, where G is
    the number of groups of points, K is the number of points, and 2 represents
    X and Y dimension coordinates
    :return: array of matrices of shape either Gx3xK or 3xK, where each
    column represents a single point in homogeneous coordinates with Z
    coordinate set to 1
    """
    remove_dim = False
    if points.ndim == 2:
        points = points[None, ...]
        remove_dim = True
    
    if points.ndim != 3:
        raise ValueError(
            f"invalid number of dimensions, expected 2 or 3, got {points.ndim}"
        )
    
    dummy_ones = np.ones((*points.shape[:-1], 1))
    points_homo = np.transpose(np.dstack((points, dummy_ones)), axes=(0, 2, 1))
    
    if remove_dim:
        points_homo = np.squeeze(points_homo)
    
    return points_homo


def _transform_points(
        points: np.ndarray, homography: np.ndarray
) -> np.ndarray:
    """Transforms a given set of points using a specific homography matrix.
    
    :param points: array of points of shape Nx2, where N is the number of
    points and 2 represents X and Y dimension coordinates
    :param homography: homography matrix to transform the points with
    :return: array pf transformed points of shape Nx2
    """
    if points.ndim != 2:
        raise ValueError(
            f"invalid number of dimensions, expected 2 got {points.ndim}")
    if points.shape[-1] != 2:
        raise ValueError("only 2D points are allowed")
    
    return np.squeeze(
        cv.perspectiveTransform(points.reshape(-1, 1, 2), homography), 1)


def _eval_reprojection_error(
        rectified_points: np.ndarray, target_points: np.ndarray
) -> float:
    """Evaluates the reprojection (reconstruction) error as L2 norm between the
    rectified and target (destination) keypoints. Simply put, computes
    a mean L2 distance between two sets of points.
    
    :param rectified_points: rectified points
    :param target_points: target (destination) points
    :return: mean L2 norm between the rectified and destination points
    """
    return cast(
        float, np.mean(np.linalg.norm(rectified_points - target_points))
    )


def _transform_points_groups_stacked(
        homography: np.ndarray, affine_matrices: np.ndarray,
        points_groups: np.ndarray
) -> np.ndarray:
    """Computes a transformations of each point group by a specific homography
    matrix followed by an affine transformation matrix. In the end, the
    transformed points are represented using homogeneous coordinates, i.e.,
    [x, y, z] becomes [x/z, y/z, 1]. For each group of points P, the global
    homography H and a local affine matrix A is used to compute the
    multiplication of three matrices "AHP".
    
    :param homography: a common homography for each point group
    :param affine_matrices: a list of affine matrices belonging to each point
    group with shape Gx3x3, where G is the number of groups
    :param points_groups: points in homogeneous coordinates in shape Gx3xK,
    where
    G is the number of groups and K is the number of keypoints
    :return:
    """
    return np.stack(
        [x / x[-1, :] for x in np.einsum(
            'gij,jk,gkl->gil', affine_matrices, homography, points_groups
        )]
    )


def _optimize_affine_matrices(
        warped_points_groups: np.ndarray, target_points: np.ndarray,
        homography: np.ndarray, ref_points_group_index: int = 0
) -> np.ndarray:
    """Finds optimal limited affine matrices with 4 DoF consisting only of
    translation, rotation and uniform scaling. Each found matrix is a mapping
    between the warped points groups and target points after being rectified by
    the given homography. In other words, each group is first projected onto a
    plane which is not subjected to perspective distortion. Then, these
    transformed points are projected again onto the target keypoints using only
    affine transformations described above. This function finds the optimal
    affine matrix for each points group.
    
    :param warped_points_groups: groups of warped keypoints of shape GxKx2,
    where G is the number of groups and K is the number of points
    :param target_points: target points onto which the affine transformation
    should project the rectified points, given by Kx2 shape
    :param homography: homography to use for each points group for rectification
    :param ref_points_group_index: index of the reference points group to which
    only an identity mapping instead of 4 DoF affine is applied
    :return: a NumPy array of affine matrices with shape Gx3x3, where G is the
    number of points groups
    """
    affine_matrices = np.empty((len(warped_points_groups), 3, 3))
    
    for i, src_points_group in enumerate(warped_points_groups):
        if i == ref_points_group_index:
            # In case of the reference points group, no affine transformation is
            # needed, so we just use the identity mapping for clarity.
            # | 1 0 0 |
            # | 0 1 0 |
            # | 0 0 1 |
            affine_matrices[i, :] = np.eye(3)
        else:
            warped_points_group_tansformed = _transform_points(
                src_points_group, homography
            )
            try:
                # Estimates a 2x3 affine transformation matrix (4 DoF):
                # | s * cos(a)   -s * sin(a)   t_x |
                # | s * sin(a)    s * cos(a)   t_y |
                affine_matrix, _ = cv.estimateAffinePartial2D(
                    warped_points_group_tansformed, target_points
                )
            except ValueError as e:
                raise TransformationNotFoundError(
                    f"affine transformation not found: {str(e)}"
                )
            else:
                # Expand the 2x3 affine matrix to 3x3 (4 DoF):
                # | a11 a12 a13 |
                # | a21 a22 a23 |
                # | 0   0   1   |
                affine_matrix = np.concatenate((affine_matrix, [(0, 0, 1)]))
                affine_matrices[i, :] = affine_matrix
    
    return affine_matrices


def _eval_points_group_score(
        warped_points_groups: np.ndarray, target_points: np.ndarray,
        homography: np.ndarray, ref_points_group_index: int
) -> float:
    """Evaluates the score function for a given reference points group. It
    computes the reprojection error for the given points group with respect to
    the specific homography.
    
    :param warped_points_groups: warped points groups
    :param target_points: target points to project the warped points onto
    :param homography: homography matrix to perform the rectification with
    :param ref_points_group_index: index of the reference points group
    :return: score computed as reprojection error
    """
    affine_matrices = _optimize_affine_matrices(
        warped_points_groups, target_points, homography, ref_points_group_index
    )
    rectified_points = _transform_points_groups_stacked(
        homography, affine_matrices,
        _convert_to_homogeneous(warped_points_groups)
    )
    score = _eval_reprojection_error(
        rectified_points, _convert_to_homogeneous(target_points[None, ...])
    )
    
    return score


def rank_homographies(
        warped_points_groups: np.ndarray, target_points_groups: np.ndarray,
        homographies: np.ndarray
) -> Sequence[int]:
    """Performs the homography ranking of multiple homographies. Each homography
    represents a mapping between i-th warped points group and i-th target points
    group. The ranking produces a list of indices that can be used to reorder
    the homographies according to their reprojection error.
    
    :param warped_points_groups: warped points groups of shape GxKx2, where G is
    the number of groups and K is the number of points
    :param target_points_groups: target points group of shape GxKx2, where G is
    the number of groups and K is the number of points
    :param homographies: a list of homographies of shape Gx3x3
    :return: a list of G integer indices representing the homography ranking
    """
    assert warped_points_groups.shape == target_points_groups.shape
    assert warped_points_groups.shape[-1] == 2
    assert warped_points_groups.ndim == 3
    assert len(warped_points_groups) == len(homographies)
    
    scores = [
        _eval_points_group_score(
            warped_points_groups, target_points_groups[i], homographies[i], i
        )
        for i in range(len(homographies))
    ]
    return tuple(np.argsort(scores))
