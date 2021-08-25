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

import numpy as np


DimensionT = Union[int, float]


class PositionGenerator(abc.ABC):
    def generate_absolute_position(
            self, width: DimensionT, height: DimensionT
    ) -> np.ndarray:
        relative_pos = self.generate_relative_position()
        return np.array((width, height), dtype=np.float) * relative_pos
    
    @abc.abstractmethod
    def generate_relative_position(self) -> np.ndarray:
        pass


class FixedPositionGen(PositionGenerator):
    def __init__(self, x_relative: float, y_relative: float) -> None:
        self.position = np.array((x_relative, y_relative), dtype=np.float)
    
    def generate_relative_position(self) -> np.ndarray:
        return self.position
