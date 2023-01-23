#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : EC_Tracking.py
@Author  : jay.zhu
@Time    : 2023/1/19 23:52
"""
from optimization.EC.EC_WithStat_Base import EC_WithStat_Base

class EC_Tracking(EC_WithStat_Base):
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, otherTerminalHandler=None, useCuda=False):
        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                         statRegisters, otherTerminalHandler, useCuda)
        self.ECDynOptHyperMutation_ECArgsDictValueController = {}
    
    def fitting(self, isOffspring=True):
        # 因为是动态的，所以最优的fitting是会变的，所以先fitting最优的那个
        self.bestChromosomesFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX], self.bestChromosomesAimFuncValue[
            self.BEST_IN_ALL_GEN_DIM_INDEX] = self.fittingOne(self.bestChromosome[:, self.BEST_IN_ALL_GEN_DIM_INDEX],
                                                              self.evalVars)

        super().fitting(isOffspring)

    def optimize(self, **kwargs):
        if kwargs.get("init"):
            self.firstRun = True
            self.ADE_firstUpdateDiffArg = True
            self.clearBestChromosome(self.BEST_IN_ALL_GEN_DIM_INDEX)
            self.clearBestChromosome(self.BEST_IN_NOW_GEN_DIM_INDEX)
        return super().optimize()
