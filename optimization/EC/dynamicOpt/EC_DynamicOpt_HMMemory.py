#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : EC_DynamicOpt_HMMemory.py
@Author  : jay.zhu
@Time    : 2022/12/28 17:47
"""
import numpy as np
import heapq
from optimization.EC.dynamicOpt.EC_DynamicOpt_HyperMutation import EC_DynamicOpt_HyperMutation

class EC_DynamicOpt_HMMemory(EC_DynamicOpt_HyperMutation):
    EC_DYNAMIC_OPT_HMMEMORY_DEFAULT_ARGS = {
        "bestArchivesMaxSize": 10,
    }
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, changeDetectorRegisters=None, otherTerminalHandler=None,
                 useCuda=False):
        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                         statRegisters, changeDetectorRegisters, otherTerminalHandler, useCuda)
        self.ECArgsDictValueController.update(self.EC_DYNAMIC_OPT_HMMEMORY_DEFAULT_ARGS)
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
        if len(self.bestArchives) < self.ECArgsDictValueController["bestArchivesMaxSize"]:
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
