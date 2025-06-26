#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : GWO_IGWO_PENG.py
@Author  : jay.zhu
@Time    : 2023/3/10 19:53
"""
import numpy as np
import optimization.common.optimizationCommonFunctions as ocf
from optimization.GWO.GWO_Base import GWO_Base

class GWO_IGWO_PENG(GWO_Base):
    __DEFAULT_IGWO_PENG_ARGS = {
        "b1": 0.7,
        "b2": 0.3,
        "selectNewPosPro": 0.4
    }

    def __init__(self, n, dimNum, positionMaxConstraint, positionMinConstraint, evalVars, optimizeWay, needEpochTimes,
                 GWOArgs, otherTerminalHandler=None, useCuda=False):
        super().__init__(n=n,
                      dimNum=dimNum,
                      positionMaxConstraint=positionMaxConstraint,
                      positionMinConstraint=positionMinConstraint,
                      evalVars=evalVars,
                      optimizeWay=optimizeWay,
                      needEpochTimes=needEpochTimes,
                      GWOArgs=GWOArgs,
                      otherTerminalHandler=otherTerminalHandler,
                      useCuda=useCuda)
        self.GWOArgsDictValueController.update(newDict=self.__DEFAULT_IGWO_PENG_ARGS, onlyAddNotExists=True)


    def updateInner(self):
        super().updateInner()
        e = float(self._nowEpochTime) / float(self.needEpochTimes)
        b1 = self.GWOArgsDictValueController["b1"]
        b2 = self.GWOArgsDictValueController["b2"]
        selectNewPosPro = self.GWOArgsDictValueController["selectNewPosPro"]

        for i in range(self.Np):
            oldPosition = np.array(self.wolfsPositions[:, i])
            for j in range(self.dimNum):
                leaderPosition = np.array([self.wolfAlphaPosition[j],
                                           self.wolfBetaPosition[j],
                                           self.wolfDeltaPosition[j], ])
                newPositionByLead = np.array([self.wolfsPositions[j, i] - (2 * e * np.random.random() - e) * (
                        2 * np.random.random() * value - self.wolfsPositions[j, i]) for value in leaderPosition])
                self.wolfsPositions[j, i] = self.limitPositionValue(b1 * np.average(newPositionByLead) + b2 * self.wolfAlphaPosition[j], j)

            newFitness, _ = ocf.fittingOne(
                solution=self.wolfsPositions[:, i],
                evalVars=self.evalVars,
                optimizeWay=self.optimizeWay,
                fittingMinDenominator=
                self.GWOArgsDictValueController[
                    "fittingMinDenominator"])

            if self.cmpFitting(newFitness, self.wolfsFittingValue[i]) < 0:
                if np.random.random() > selectNewPosPro:
                    self.wolfsPositions[:, i] = oldPosition

