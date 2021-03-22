#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Milan Ondrašovič <milan.ondrasovic@gmail.com>
#
# An implementation of the "homography ranking" algorithm. The purpose of this
# script is to provide a stand-alone application that can be used to demonstrate
# the workings of the proposed method. This method is capable of ranking
# multiple different homographies that belong to geometrically similar objects
# that lie on the same plane for which we want to compute the bird's-eye view.
# We call objects as "geometrically similar" if a limited 4 DoF affine
# transformation consisting of only translation, rotation and uniform scaling
# exists between them. For more information, read the references research paper.
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
import click

from typing import Optional, cast, Sequence

import numpy as np
import cv2 as cv

from transform.homographyranking import rank_homographies


def estimate_and_rank_homographies(
        warped_points_groups: np.ndarray, target_points_groups: np.ndarray,
        homographies: Optional[np.ndarray]) -> Sequence[int]:
    if homographies is None:
        homographies = []
        
        for warped_points, target_points in zip(
                warped_points_groups, target_points_groups):
            homography, _ = cv.findHomography(
                warped_points, target_points, method=cv.RANSAC)
            homographies.append(homography)
    
    homographies = np.asarray(homographies)
    
    ranking = rank_homographies(
        warped_points_groups, target_points_groups, homographies)
    
    return ranking

@click.command()
@click.argument("warped_points_path", type=click.Path())
@click.argument("target_points_path", type=click.Path())
@click.option(
    "-o", "--output-path", type=click.Path(),
    help="Output file to save the ordering to as comma-separated integers.")
@click.option(
    "-h", "--homographies", type=click.Path(),
    help="NumPy file containing homographies as Gx3x3 floating-point tensor,"
         "where G is the same as in the WARPED_PTS shape GxKx2.")
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Enables verbose mode to print more information.")
def main(
        warped_points_path: click.Path, target_points_path: click.Path,
        output_path: click.Path, homographies: Optional[click.Path],
        verbose: bool) -> int:
    """
    This application is licensed under the MIT license.
    
    Copyright (c) 2021 Milan Ondrasovic
    
    Executes the homography ranking algorithm. Let G be the number of groups of
    point correspondences and K be the number of points (object keypoints).
    The warped (source) and target (destination) points (keypoints) are
    contained in the WARPED_POINTS_PATH and TARGET_POINTS_PATH file,
    respectively. Each file has to provide a floating-point NumPy tensor of
    shape GxKx2 and Kx2, respectively, establishing the many-to-one point
    correspondence.
    
    Homographies can either be specified in a NumPy file or estimated from
    scratch. If already estimated (when -h, --homographies is specified), then
    the HOMOGRAPHIES_FILE must contain a floating-point tensor of shape Gx3x3.
    In this context, G represents the number of homographies. Each i-th
    homography is responsible for the mapping between the i-th group of warped
    keypoints and the target keypoints. If the estimation should be performed
    by the application (when -h, --homographies is not specified), then the same
    point correspondence described above holds.
    
    The resulting order of homographies, as well as points groups, is saved as a
    text file (given by -o, --output) containing comma-separated integers
    representing the ordering indices from the "best" to "worst" as induced by
    the homography ranking algorithm. These indices can be then used as an
    output of an indirect sort to re-index the array as needed.
    """
    warped_points = np.load(cast(str, warped_points_path))
    target_points = np.load(cast(str, target_points_path))
    
    target_points = np.repeat(
        target_points[None, ...], repeats=len(warped_points), axis=0)
    
    ranking = estimate_and_rank_homographies(
        warped_points, target_points, homographies)
    indices = ",".join(map(str, ranking))
    
    if output_path:
        with open(cast(str, output_path), "wt") as out_file:
            out_file.write(indices + "\n")
    else:
        print(indices)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
