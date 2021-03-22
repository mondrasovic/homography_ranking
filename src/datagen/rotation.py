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

from common.dtypes import RangeT
from common.utils import rand_uniform
from transform.base import HomographyTransformer
from transform.rotation import Rotation3DTransformer


class Rotation3DGenerator:
    """
    A generator that randomly initiates a 3D rotation transformer.
    """
    
    def __init__(
            self, x_axis_range: RangeT, y_axis_range: RangeT,
            z_axis_range: RangeT) -> None:
        """
        Constructor.

        :param x_axis_range: range for the random uniform rotation in the x-axis
        :param y_axis_range: range for the random uniform rotation in the y-axis
        :param z_axis_range: range for the random uniform rotation in the z-axis
        """
        self.x_axis_range: RangeT = x_axis_range
        self.y_axis_range: RangeT = y_axis_range
        self.z_axis_range: RangeT = z_axis_range
    
    def build_transformer(self) -> HomographyTransformer:
        """
        Randomly initializes a 3D rotation transformer.

        :return: a 3D rotation transformer
        """
        theta = rand_uniform(*self.x_axis_range)
        phi = rand_uniform(*self.y_axis_range)
        gamma = rand_uniform(*self.z_axis_range)
        
        return Rotation3DTransformer(theta, phi, gamma)
