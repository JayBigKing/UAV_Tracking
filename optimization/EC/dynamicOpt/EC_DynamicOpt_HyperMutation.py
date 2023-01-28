#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : EC_DynamicOpt_HyperMutation.py
@Author  : jay.zhu
@Time    : 2022/10/11 22:58
"""
import random
import numpy as np
from optimization.common.ArgsDictValueController import ArgsDictValueController
from optimization.EC.dynamicOpt.EC_DynamicOpt_Base import EC_DynamicOpt_Base
from optimization.EC.dynamicOpt.EC_ChangeDetect import EC_ChangeDetector_PerformanceThresh


class EC_DynamicOpt_HyperMutation(EC_DynamicOpt_Base):
    EC_DYNAMIC_OPT_HYPER_MUTATION_DEFAULT_CHANGE_DETECTOR_REG_DICT = {
        "mutationProbabilityWhenChange": 0.5,
        "mutationProbabilityWhenNormal": 0.05,
        "performanceThreshold": 3
    }

    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, changeDetectorRegisters=None, otherTerminalHandler=None,
                 useCuda=False):
        # if statRegisters is None:
        #     statRegisters = [self.EC_DynamicOpt_HyperMutation_PrintOutEveryGenFunc]
        # else:
        #     statRegisters.append(self.EC_DynamicOpt_HyperMutation_PrintOutEveryGenFunc)

        if changeDetectorRegisters is None:
            if ECArgs.get("performanceThreshold"):
                changeDetectorRegisters = EC_ChangeDetector_PerformanceThresh(n, {
                    "performanceThreshold": ECArgs["performanceThreshold"], "fittingCmpFunc": self.cmpFitting})
            else:
                raise ValueError("ECArgs need a key-value pair which key named performanceThreshold")

        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay,
                         needEpochTimes, ECArgs, statRegisters, changeDetectorRegisters,
                         otherTerminalHandler, useCuda)
        self.ECDynOptHyperMutation_ECArgsDictValueController = ArgsDictValueController(userArgsDict=ECArgs,
                                                                                       defaultArgsDict=self.EC_DYNAMIC_OPT_HYPER_MUTATION_DEFAULT_CHANGE_DETECTOR_REG_DICT,
                                                                                       onlyUseDefaultKey=True)



    def crossover(self):
        mutationChromosomes = np.array(self.middleChromosomes)
        # alpha = self.ECArgsDictValueController["floatCrossoverAlpha"]
        # alphaRemain = 1. - alpha
        # for i in range(self.Np):
        #     r1, r2 = random.sample(range(self.Np), 2)
        #     for j in range(self.dimNum):
        #         self.middleChromosomes[j][i] = alpha * mutationChromosomes[j][r2] + \
        #                                        alphaRemain * mutationChromosomes[j][r1]
        middleChroIndex = 0
        for i in range(int(self.Np / 2)):
            r1, r2 = random.sample(range(self.Np), 2)
            crossoverPos = random.randint(0, self.dimNum)

            for j in range(0, crossoverPos):
                self.middleChromosomes[j][middleChroIndex] = mutationChromosomes[j][r2]
                if middleChroIndex + 1 < self.Np:
                    self.middleChromosomes[j][middleChroIndex + 1] = mutationChromosomes[j][r1]

            for j in range(crossoverPos, self.dimNum):
                self.middleChromosomes[j][middleChroIndex] = mutationChromosomes[j][r1]
                if middleChroIndex + 1 < self.Np:
                    self.middleChromosomes[j][middleChroIndex + 1] = mutationChromosomes[j][r2]

            middleChroIndex += 2

    def adaptToEnvironmentWhenNormal(self):
        self.ECArgsDictValueController["mutationProbability"] = self.ECDynOptHyperMutation_ECArgsDictValueController[
            "mutationProbabilityWhenNormal"]

    def adaptToEnvironmentWhenChange(self):
        super().adaptToEnvironmentWhenChange()
        self.ECArgsDictValueController["mutationProbability"] = self.ECDynOptHyperMutation_ECArgsDictValueController[
            "mutationProbabilityWhenChange"]

    def adaptToEnvironmentWhenRefractoryPeriod(self):
        refractoryPeriodLength = self.ECArgsDictValueController["refractoryPeriodLength"]
        if self.refractoryPeriodTick == refractoryPeriodLength:
            self.ECArgsDictValueController["mutationProbability"] = \
                self.ECDynOptHyperMutation_ECArgsDictValueController["mutationProbabilityWhenNormal"]

    def EC_DynamicOpt_HyperMutation_PrintOutEveryGenFunc(self, **kwargs):
        bestAimFuncVal = kwargs["chromosomesAimFuncValue"]
        bestFittingFuncVal = kwargs["chromosomesFittingValue"]
        print('[BestAimFuncVal, BestFittingVal, bestChromosome]: [%f, %f, %r]' % (bestAimFuncVal, bestFittingFuncVal, self.ECArgsDictValueController["mutationProbability"]))