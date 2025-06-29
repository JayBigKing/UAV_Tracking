#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : GWO_IGWO_PENGTracking.py
@Author  : jay.zhu
@Time    : 2023/3/10 20:11
"""
import optimization.common.optimizationCommonFunctions as ocf
from optimization.GWO.GWO_IGWO_PENG import GWO_IGWO_PENG


class GWO_IGWO_PENGTracking(GWO_IGWO_PENG):
    # def __init__(self, n, dimNum, positionMaxConstraint, positionMinConstraint, evalVars, optimizeWay, needEpochTimes,
    #              GWOArgs, otherTerminalHandler=None, useCuda=False):
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 otherTerminalHandler=None, useCuda=False):
        super().__init__(n=n,
                         dimNum=dimNum,
                         positionMaxConstraint=maxConstraint,
                         positionMinConstraint=minConstraint,
                         evalVars=evalVars,
                         optimizeWay=otimizeWay,
                         needEpochTimes=needEpochTimes,
                         GWOArgs=ECArgs,
                         otherTerminalHandler=otherTerminalHandler,
                         useCuda=useCuda)

        self.ECArgsDictValueController = self.GWOArgsDictValueController
        self.DEOArgDictValueController = {}

    def optimize(self, **kwargs):
        if kwargs.get("init"):
            self.firstRun = True
            self.clearBestWolfPosition(self.BEST_IN_ALL_GEN_DIM_INDEX)
            self.clearBestWolfPosition(self.BEST_IN_NOW_GEN_DIM_INDEX)
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
            solution=self.globalBestWolfPosition[:, self.BEST_IN_ALL_GEN_DIM_INDEX],
            evalVars=self.evalVars,
            optimizeWay=self.optimizeWay,
            fittingMinDenominator=
            self.GWOArgsDictValueController[
                "fittingMinDenominator"])

        super().fitting()
