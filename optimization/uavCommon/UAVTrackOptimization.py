#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAVTrackOptimization.py
@Author  : jay.zhu
@Time    : 2022/12/16 1:06
"""


class UAVTrackOptimization:
    def __init__(self, ECArgsDictValueController = None, ECDynOptHyperMutation_ECArgsDictValueController = None):
        if ECArgsDictValueController is None:
            self.ECArgsDictValueController = {}
        else:
            self.ECArgsDictValueController = ECArgsDictValueController

        if ECDynOptHyperMutation_ECArgsDictValueController is None:
            self.ECDynOptHyperMutation_ECArgsDictValueController = {}
        else:
            self.ECDynOptHyperMutation_ECArgsDictValueController = ECDynOptHyperMutation_ECArgsDictValueController
