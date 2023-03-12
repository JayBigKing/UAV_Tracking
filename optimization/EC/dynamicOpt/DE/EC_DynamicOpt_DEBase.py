#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : EC_DynamicOpt_DEBase.py
@Author  : jay.zhu
@Time    : 2023/2/19 10:57
"""
from optimization.EC.dynamicOpt.EC_DynamicOpt_Base import EC_DynamicOpt_Base
from optimization.EC.DiffEC.EC_DiffEC_Base import EC_DiffEC_Base

class EC_DynamicOpt_DEBase(EC_DynamicOpt_Base, EC_DiffEC_Base):
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, changeDetectorRegisters="EvaluateSolutions", otherTerminalHandler=None,
                 useCuda=False):
        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                         statRegisters, changeDetectorRegisters, otherTerminalHandler,
                         useCuda)



