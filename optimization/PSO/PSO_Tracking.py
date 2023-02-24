#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : PSO_Tracking.py
@Author  : jay.zhu
@Time    : 2022/12/15 18:53
"""
from optimization.PSO.PSO_Base import PSO_Base
import optimization.common.optimizationCommonFunctions as ocf


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
        self.DEOArgDictValueController = {}

    def optimize(self, **kwargs):
        if kwargs.get("init"):
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

    def fitting(self):
        # 因为是动态的，所以最优的fitting是会变的，所以先fitting最优的那个
        self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX], self.globalBestAimFuncValue[
            self.BEST_IN_ALL_GEN_DIM_INDEX] = ocf.fittingOne(
                solution=self.globalBestParticlePosition[:, self.BEST_IN_ALL_GEN_DIM_INDEX],
                evalVars=self.evalVars,
                optimizeWay=self.optimizeWay,
                fittingMinDenominator=
                self.PSOArgsDictValueController[
                    "fittingMinDenominator"])

        super().fitting()