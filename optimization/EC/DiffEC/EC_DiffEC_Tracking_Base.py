#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : EC_DiffEC_Tracking_Base.py
@Author  : jay.zhu
@Time    : 2022/12/16 2:19
"""
from optimization.EC.DiffEC.EC_DiffEC_Base import EC_DiffEC_Base

class EC_DiffEC_Tracking_Base(EC_DiffEC_Base):
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, otherTerminalHandler=None, useCuda=False):

        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                         statRegisters, otherTerminalHandler, useCuda)

        self.ECDynOptHyperMutation_ECArgsDictValueController = {}


    def optimize(self):
        self.firstRun = True
        self.ADE_firstUpdateDiffArg = True
        self.clearBestChromosome(self.BEST_IN_ALL_GEN_DIM_INDEX)
        self.clearBestChromosome(self.BEST_IN_NOW_GEN_DIM_INDEX)
        return super().optimize()
