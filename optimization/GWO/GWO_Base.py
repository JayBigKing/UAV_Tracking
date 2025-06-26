#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : GWO_Base.py
@Author  : jay.zhu
@Time    : 2023/3/10 15:54
"""
import random
import numpy as np
from algorithmTool.sortTool.HeapSort import HeapSort
from optimization.common.optimizationCommonEnum import OptimizationWay
from optimization.common.ArgsDictValueController import ArgsDictValueController
import optimization.common.optimizationCommonFunctions as ocf


class GWO_Base:
    PERSON_NOW_FITNESS_INDEX = 0
    PERSON_BEST_FITNESS_INDEX = 1

    BEST_IN_ALL_GEN_DIM_INDEX = 0
    BEST_IN_NOW_GEN_DIM_INDEX = 1

    __DEFAULT_GWO_BASE_ARGS = {
        "fittingMinDenominator": 0.2,
        "updateInnerEpochTimes": 1,
        "borders": [0, 1],
    }

    def __init__(self, n, dimNum, positionMaxConstraint, positionMinConstraint, evalVars, optimizeWay, needEpochTimes,
                 GWOArgs, otherTerminalHandler=None, useCuda=False):
        self.Np = n
        self.dimNum = dimNum
        self.positionMaxConstraint = positionMaxConstraint
        self.positionMinConstraint = positionMinConstraint
        self.GWOArgsDictValueController = ArgsDictValueController(GWOArgs, self.__DEFAULT_GWO_BASE_ARGS)

        self.evalVars = evalVars
        if isinstance(optimizeWay, OptimizationWay) is False:
            raise TypeError("optimization way should be an instance of enum OptimizationWay")
        else:
            self.optimizeWay = optimizeWay

        self.wolfsPositions = np.zeros((dimNum, n))
        self.globalBestWolfPosition = np.zeros((dimNum, 2))
        self.wolfAlphaPosition = np.zeros(dimNum)
        self.wolfBetaPosition = np.zeros(dimNum)
        self.wolfDeltaPosition = np.zeros(dimNum)

        self.wolfsFittingValue = np.zeros(n)
        self.wolfsAimFuncValue = np.zeros(n)
        self.globalBestFittingValue = np.zeros(2)
        self.globalBestAimFuncValue = np.zeros(2)
        # self.wolfAlphaFittingValue = 0.
        # self.wolfBetaFittingValue = 0.
        # self.wolfDeltaFittingValue = 0.
        # self.wolfAlphaAimFuncValue = 0.
        # self.wolfBetaAimFuncValue = 0.
        # self.wolfDeltaAimFuncValue = 0.

        self.wolfAlphaIndex = 0
        self.wolfBetaIndex = 0
        self.wolfDeltaIndex = 0

        self._nowEpochTime = 0
        self.needEpochTimes = needEpochTimes
        self.updateInnerEpochTimes = self.GWOArgsDictValueController["updateInnerEpochTimes"]

        self.GWOArgs = self.GWOArgsDictValueController.userArgsDict
        self.borders = self.GWOArgs["borders"]  # 如果没有self.ECArgs["borders"] 的话，self.borders就是[]
        self.otherTerminalHandler = otherTerminalHandler
        self.useCuda = useCuda

        self.firstRun = True

    def wolfInit(self):
        for i in range(self.Np):
            self.wolfsFittingValue[i], self.wolfsAimFuncValue[i] = 0., 0

        for i in range(0, self.Np):
            for j in range(0, self.dimNum):
                self.wolfsPositions[j, i] = self.positionMinConstraint[
                                                j] + np.random.random() * (
                                                    self.positionMaxConstraint[j] -
                                                    self.positionMinConstraint[j])

    def initShouldContinueVar(self, otherTerminalHandler=None):
        if otherTerminalHandler is not None:
            otherTerminalHandler(initFlag=True)
        self._nowEpochTime = 0

    def optimize(self):
        return self.optimization()

    def optimization(self):
        if self.firstRun == True:
            self.wolfInit()
            self.fitting()
            self.firstRun = False
        self.initShouldContinueVar(self.otherTerminalHandler)
        while self.shouldContinue(self.otherTerminalHandler):
            self.optimizeInner()

        return np.array(self.globalBestWolfPosition[:, self.BEST_IN_ALL_GEN_DIM_INDEX]), \
               self.globalBestAimFuncValue[self.BEST_IN_ALL_GEN_DIM_INDEX], \
               self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX]

    def optimizeInner(self):
        self.clearBestWolfPosition(self.BEST_IN_NOW_GEN_DIM_INDEX)
        self.update()
        self.fitting()

    def update(self):
        for innerEpoch in range(self.updateInnerEpochTimes):
            self.updateInner()

    def updateInner(self):
        e = float(self._nowEpochTime) / float(self.needEpochTimes)
        for i in range(self.Np):
            for j in range(self.dimNum):
                leaderPosition = np.array([self.wolfAlphaPosition[j],
                                           self.wolfBetaPosition[j],
                                           self.wolfDeltaPosition[j], ])
                newPositionByLead = np.array([self.wolfsPositions[j, i] - (2 * e * np.random.random() - e) * (
                        2 * np.random.random() * value - self.wolfsPositions[j, i]) for value in leaderPosition])
                self.wolfsPositions[j, i] = self.limitPositionValue(np.average(newPositionByLead), j)

    def fitting(self):
        for i in range(0, self.Np):
            self.wolfsFittingValue[i], self.wolfsAimFuncValue[i] = ocf.fittingOne(
                solution=self.wolfsPositions[:, i],
                evalVars=self.evalVars,
                optimizeWay=self.optimizeWay,
                fittingMinDenominator=
                self.GWOArgsDictValueController[
                    "fittingMinDenominator"])
        self.updateLeaderWolfs()
        self.cmpToBestPositionAndStore(self.wolfAlphaIndex)

    def cmpFitting(self, val1, val2):
        return (val1 - val2)

    def shouldContinue(self, otherTerminalHandler):
        shouldContinueFlag, self._nowEpochTime = ocf.shouldContinue(nowEpochTime=self._nowEpochTime,
                                                                    bestFittingValue=self.globalBestFittingValue[
                                                                        self.BEST_IN_ALL_GEN_DIM_INDEX],
                                                                    bestAimFuncValue=self.globalBestAimFuncValue[
                                                                        self.BEST_IN_ALL_GEN_DIM_INDEX],
                                                                    needEpochTimes=self.needEpochTimes,
                                                                    otherTerminalHandler=otherTerminalHandler)
        return shouldContinueFlag

    def updateLeaderWolfs(self):
        heapSort = HeapSort()
        leaderWolfsIndex = heapSort.getTopNInSortIndexList(dataList=self.wolfsFittingValue,
                                                           n=3,
                                                           sortWay=1,
                                                           cmp=self.cmpFitting)
        self.wolfAlphaIndex, self.wolfBetaIndex, self.wolfDeltaIndex = leaderWolfsIndex[0], leaderWolfsIndex[1], \
                                                                       leaderWolfsIndex[2]
        self.wolfAlphaPosition, self.wolfBetaPosition, self.wolfDeltaPosition = np.array(
            self.wolfsPositions[:, self.wolfAlphaIndex]), np.array(self.wolfsPositions[:, self.wolfBetaIndex]), np.array(
            self.wolfsPositions[:, self.wolfDeltaIndex])

    def clearBestWolfPosition(self, whichBestWolf):
        if whichBestWolf == self.BEST_IN_NOW_GEN_DIM_INDEX:
            self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX], self.globalBestAimFuncValue[
                self.BEST_IN_NOW_GEN_DIM_INDEX] = 0., 0.
        elif whichBestWolf == self.BEST_IN_ALL_GEN_DIM_INDEX:
            self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX], self.globalBestAimFuncValue[
                self.BEST_IN_ALL_GEN_DIM_INDEX] = 0., 0.

    def cmpToBestPositionAndStore(self, particleIndex):
        if self.cmpFitting(self.wolfsFittingValue[particleIndex],
                           self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX]) > 0:
            self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX] = self.wolfsFittingValue[particleIndex]
            self.globalBestAimFuncValue[self.BEST_IN_NOW_GEN_DIM_INDEX] = self.wolfsAimFuncValue[particleIndex]
            self.globalBestWolfPosition[:, self.BEST_IN_NOW_GEN_DIM_INDEX] = np.array(
                self.wolfsPositions[:, particleIndex])

            if self.cmpFitting(self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX],
                               self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX]) > 0:
                self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX], self.globalBestAimFuncValue[
                    self.BEST_IN_ALL_GEN_DIM_INDEX] = self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX], \
                                                      self.globalBestAimFuncValue[self.BEST_IN_NOW_GEN_DIM_INDEX]
                self.globalBestWolfPosition[:, self.BEST_IN_ALL_GEN_DIM_INDEX] = np.array(
                    self.globalBestWolfPosition[:, self.BEST_IN_NOW_GEN_DIM_INDEX])

    def limitPositionValue(self, positionValue, dimIndex):
        return ocf.limitValue(positionValue, dimIndex, self.positionMaxConstraint, self.positionMinConstraint,
                              self.borders)
