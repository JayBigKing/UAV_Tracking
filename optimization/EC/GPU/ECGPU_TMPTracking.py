#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : ECGPU_TMPTracking.py
@Author  : jay.zhu
@Time    : 2023/2/26 21:32
"""
from optimization.EC.GPU.ECGPU_Base import ECGPU_Base

class ECGPU_TMPTracking(ECGPU_Base):
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, otherTerminalHandler=None, useCuda=True):
        super().__init__(n=n,
                         dimNum=dimNum,
                         maxConstraint=maxConstraint,
                         minConstraint=minConstraint,
                         evalVars=evalVars,
                         otimizeWay=otimizeWay,
                         needEpochTimes=needEpochTimes,
                         ECArgs=ECArgs,
                         statRegisters=statRegisters,
                         otherTerminalHandler=otherTerminalHandler,
                         useCuda=useCuda)

        self.DEOArgDictValueController = {}

    def optimize(self, **kwargs):
        if kwargs.get("init"):
            self.firstRun = True
            self.clearBestChromosome(self.BEST_IN_ALL_GEN_DIM_INDEX)
            self.clearBestChromosome(self.BEST_IN_NOW_GEN_DIM_INDEX)
        return super().optimize()

