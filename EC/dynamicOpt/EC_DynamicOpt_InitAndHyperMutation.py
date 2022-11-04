#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : EC_DynamicOpt_InitAndHyperMutation.py
@Author  : jay.zhu
@Time    : 2022/11/3 21:38
"""
from EC.dynamicOpt.EC_DynamicOpt_HyperMutation import EC_DynamicOpt_HyperMutation


class EC_DynamicOpt_InitAndHyperMutation(EC_DynamicOpt_HyperMutation):
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, changeDetectorRegisters=None, otherTerminalHandler=None,
                 useCuda=False):
        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                         statRegisters, changeDetectorRegisters, otherTerminalHandler, useCuda)

    def adaptToEnvironmentWhenChange(self):
        super().adaptToEnvironmentWhenChange()
        self.chromosomeInit()
        self.fitting(isOffspring=False)
