#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : PSO_Tracking.py
@Author  : jay.zhu
@Time    : 2022/12/15 18:53
"""
from optimization.PSO.PSO_Base import PSO_Base


class PSO_Tracking(PSO_Base):
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes,
                 ECArgs, velocityMaxConstraint=None, velocityMinConstraint=None, otherTerminalHandler=None,
                 useCuda=False):
        super().__init__(n = n,
                         dimNum=dimNum,
                         positionMaxConstraint=maxConstraint,
                         positionMinConstraint=minConstraint,
                         evalVars=evalVars,
                         optimizeWay=otimizeWay,
                         needEpochTimes=needEpochTimes,
                         PSOArgs=ECArgs,
                         velocityMaxConstraint=velocityMaxConstraint,
                         velocityMinConstraint=velocityMinConstraint,
                         otherTerminalHandler=otherTerminalHandler,
                         useCuda=useCuda)

        self.ECArgsDictValueController = self.PSOArgsDictValueController
        self.ECDynOptHyperMutation_ECArgsDictValueController = {}

    def optimize(self):
        self.firstRun = True
        self.clearBestChromosome(self.BEST_IN_ALL_GEN_DIM_INDEX)
        self.clearBestChromosome(self.BEST_IN_NOW_GEN_DIM_INDEX)
        return self.optimization()

    def shouldContinue(self, otherTerminalHandler=None):
        self._nowEpochTime += 1
        if otherTerminalHandler is not None:
            if otherTerminalHandler(
                    bestChromosomesFittingValue=self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX]) is False:
                return False

        if self._nowEpochTime <= self.needEpochTimes:
            return True
        else:
            return False
