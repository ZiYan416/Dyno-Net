# -*- coding: utf-8 -*-
# @Author  : Haonan Wang
# @File    : CTrans.py
# @Software: PyCharm
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

__all__ = ['GetIndexOutput']

class GetIndexOutput(nn.Module):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index
    
    def forward(self, x):
        return x[self.index]