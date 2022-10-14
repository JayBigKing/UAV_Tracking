#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : mathFunction.py
@Author  : jay.zhu
@Time    : 2022/10/14 12:32
"""
import numpy as np

def Jay_sigmoid(x):
    return 1 / (1 + np.exp(-x))
