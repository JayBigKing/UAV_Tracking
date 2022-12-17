#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : EC_WithStat_Base.py
@Author  : jay.zhu
@Time    : 2022/10/9 15:00
"""

import numpy as np
from optimization.EC.EC_Base import EC_Base
from dataStatistics.statFuncListGenerator import statFuncListGenerator


class EC_WithStat_Base(EC_Base):
    __DEFAULT_EC_STAT_FUNC_STR = ["bestOverGen", "HammingDis", "InertiaDis", "ARR", "printOutEveryGen"]

    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, otherTerminalHandler=None, useCuda=False):
        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                         otherTerminalHandler, useCuda)

        self.__DEFAULT_EC_STAT_FUNC_MAP = {
            "bestOverGen": self.EC_WithStat_BestOverGenFunc,
            "HammingDis": self.EC_WithStat_HammingDisFunc,
            "InertiaDis": self.EC_WithStat_InertiaDisFunc,
            "printOutEveryGen" : self.EC_WithStat_PrintOutEveryGenFunc,
        }
        # self.statFuncReg = []
        # for item in statRegisters:
        #     if isinstance(item, str):
        #         self.statFuncReg.append(self.__DEFAULT_EC_STAT_FUNC_MAP[item])
        #     elif isfunction(item):
        #         self.statFuncReg.append(item)
        self.statFuncReg = statFuncListGenerator(statRegisters, self.__DEFAULT_EC_STAT_FUNC_MAP)

        self.statBaseInit()

    def statBaseInit(self):
        self.EC_WithStat_bestChromosomesOverGen = []
        self.EC_WithStat_avgOfBestChromosomesVal = [0., 0.]

        self.EC_WithStat_HammingDis = []
        self.EC_WithStat_InertiaDis = []
        self.EC_WithStat_ARR = 0.

    def EC_WithStat_callStatFunc(self):
        for item in self.statFuncReg:
            item(chromosomes=self.chromosomes[self.BEST_IN_ALL_GEN_DIM_INDEX],
                 chromosomesAimFuncValue=self.bestChromosomesAimFuncValue[self.BEST_IN_ALL_GEN_DIM_INDEX],
                 chromosomesFittingValue=self.bestChromosomesFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX],)

    def EC_WithStat_BestOverGenFunc(self, **kwargs):
        bestAimFuncVal = kwargs["chromosomesAimFuncValue"]
        bestFittingFuncVal = kwargs["chromosomesFittingValue"]
        self.EC_WithStat_bestChromosomesOverGen.append([bestAimFuncVal, bestFittingFuncVal])

    def EC_WithStat_HammingDisFunc(self, **kwargs):
        pass

    def EC_WithStat_InertiaDisFunc(self, **kwargs):
        chromosomes = kwargs["chromosomes"]
        centroid = np.vstack(np.sum(self.chromosomes, axis=1) / self.Np)
        self.EC_WithStat_InertiaDis.append(np.sum(np.square(chromosomes - centroid)))

    def EC_WithStat_ARRFunc(self, **kwargs):
        pass

    def EC_WithStat_PrintOutEveryGenFunc(self, **kwargs):
        bestAimFuncVal = kwargs["chromosomesAimFuncValue"]
        bestFittingFuncVal = kwargs["chromosomesFittingValue"]
        print('[BestAimFuncVal, BestFittingVal, bestChromosome]: [%f, %f, %r]' % (bestAimFuncVal, bestFittingFuncVal, self.ECArgsDictValueController["mutationProbability"]))

    # reload base class function
    def optimizeInner(self):
        super().optimizeInner()
        self.EC_WithStat_callStatFunc()

    # interface function
    def EC_WithStat_GetBestOverGen(self):
        return np.average(np.array(self.EC_WithStat_bestChromosomesOverGen), axis=0), np.array(
            self.EC_WithStat_bestChromosomesOverGen)
