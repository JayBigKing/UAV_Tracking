#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : EC_DynamicOpt_DEMemory.py
@Author  : jay.zhu
@Time    : 2023/2/19 11:52
"""
import numpy as np
import heapq
from optimization.common.ArgsDictValueController import ArgsDictValueController
from optimization.EC.dynamicOpt.EC_ChangeDetect import EC_ChangeDetector_PerformanceThresh
from optimization.EC.dynamicOpt.DE.EC_DynamicOpt_DEBase import EC_DynamicOpt_DEBase


class EC_DynamicOpt_DEMemory(EC_DynamicOpt_DEBase):
    EC_DYNAMIC_OPT_DE_MEMORY_DEFAULT_CHANGE_DETECTOR_REG_DICT = {
        "performanceThreshold": 3,
        "bestArchivesMaxSize": 10,
    }

    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, changeDetectorRegisters=None, otherTerminalHandler=None,
                 useCuda=False):

        if changeDetectorRegisters is None:
            if ECArgs.get("performanceThreshold"):
                changeDetectorRegisters = EC_ChangeDetector_PerformanceThresh(n, {
                    "performanceThreshold": ECArgs["performanceThreshold"], "fittingCmpFunc": self.cmpFitting})
            else:
                changeDetectorRegisters = EC_ChangeDetector_PerformanceThresh(n, {
                    "performanceThreshold": self.EC_DYNAMIC_OPT_DE_MEMORY_DEFAULT_CHANGE_DETECTOR_REG_DICT[
                        "performanceThreshold"], "fittingCmpFunc": self.cmpFitting})

        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay,
                         needEpochTimes, ECArgs, statRegisters, changeDetectorRegisters,
                         otherTerminalHandler, useCuda)

        self.DEOArgDictValueController.update(newDict=self.EC_DYNAMIC_OPT_DE_MEMORY_DEFAULT_CHANGE_DETECTOR_REG_DICT, newUserDict=ECArgs)
        self.bestArchives = []

    def select(self):
        super().select()
        replaceNum = len(self.bestArchives)
        chromosomeFitness = [item[self.CHROMOSOME_DIM_INDEX] for item in self.chromosomesFittingValue]
        if len(chromosomeFitness) == len(list(set(chromosomeFitness))):
            theLowFitness = heapq.nsmallest(replaceNum, chromosomeFitness)
            replaceIndex = list(map(chromosomeFitness.index, theLowFitness))
            for index, item in enumerate(replaceIndex):
                self.chromosomes[:, item] = np.array(self.bestArchives[index])
                self.chromosomesFittingValue[item][self.CHROMOSOME_DIM_INDEX], self.chromosomesAimFuncValue[item][
                    self.CHROMOSOME_DIM_INDEX] = self.fittingOne(
                    self.chromosomes[:, item], self.evalVars)




    def updateBestArchives(self):
        if len(self.bestArchives) < self.DEOArgDictValueController["bestArchivesMaxSize"]:
            self.bestArchives.append(np.array(self.bestChromosome[:, self.BEST_IN_ALL_GEN_DIM_INDEX]))
        else:
            centreOfArch = np.array([np.average([item[dim] for item in self.bestArchives]) for dim in range(len(self.bestArchives[0]))])
            diffFromCenter = [abs(np.sum(item - centreOfArch)) for item in self.bestArchives]
            mostSimilarIndex = 0
            for index in range(1, len(diffFromCenter)):
                if diffFromCenter[index] < diffFromCenter[mostSimilarIndex]:
                    mostSimilarIndex = index

            self.bestArchives[mostSimilarIndex] = np.array(self.bestChromosome[:, self.BEST_IN_ALL_GEN_DIM_INDEX])


    def adaptToEnvironmentWhenChange(self):
        super().adaptToEnvironmentWhenChange()

        self.updateBestArchives()
        self.chromosomeInit()
        self.clearBestChromosome(self.BEST_IN_NOW_GEN_DIM_INDEX)
        self.clearBestChromosome(self.BEST_IN_ALL_GEN_DIM_INDEX)
        self.fitting(isOffspring=False)
