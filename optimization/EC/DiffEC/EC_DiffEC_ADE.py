#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : EC_DiffEC_ADE.py
@Author  : jay.zhu
@Time    : 2022/12/15 0:54
"""
import numpy as np
import random
from optimization.EC.DiffEC.EC_DiffEC_Base import EC_DiffEC_Base,EC_CodingType

class EC_DiffEC_ADE(EC_DiffEC_Base):
    __DIFF_EC_ADE_DEFAULT_ARGS = {
        "DiffCR0": 0.1,
        "DiffCR1": 0.6,
        "DiffF0": 0.1,
        "DiffF1": 0.6,
    }
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, otherTerminalHandler=None, useCuda=False):
        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                         statRegisters, otherTerminalHandler, useCuda)
        self.ADE_firstUpdateDiffArg = True

        self.ECArgsDictValueController.update(newDict=self.__DIFF_EC_ADE_DEFAULT_ARGS)
        self.chromosomeFg = self.ECArgsDictValueController["DiffF0"]

    def mutation(self):
        mutationProbability = self.ECArgsDictValueController["mutationProbability"]
        for i in range(0, self.Np):
            r1, r2, r3 = random.sample(range(self.Np), 3)
            for j in range(0, self.dimNum):
                if self.EC_Base_codingType.value == EC_CodingType.BINARY_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.GRAY_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.FLOAT_CODING.value:
                    # if np.random.random() < mutationProbability:
                    self.middleChromosomes[j, i] = min(
                        max(self.chromosomes[j, r1] + self.chromosomeFg * (
                                self.chromosomes[j, r2] - self.chromosomes[j, r3]),
                            self.minConstraint[j]),
                        self.maxConstraint[j])
                # else:
                #     self.middleChromosomes[j, i] = self.chromosomes[j, i]
                elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
                    pass


    def updateDiffArg(self):
        super().updateDiffArg()
        valDim = self.CHROMOSOME_DIM_INDEX
        if self.ADE_firstUpdateDiffArg is True:
            self.ADE_firstUpdateDiffArg = False
            self.zeroFitness = np.sum(self.chromosomesFittingValue[:, valDim])
            self.chromosomeFg = self.ECArgsDictValueController["DiffF1"]
        else:
            F0 = self.ECArgsDictValueController["DiffF0"]
            F1 = self.ECArgsDictValueController["DiffF1"]
            nowFitness = np.sum(self.chromosomesFittingValue[:, valDim])
            self.chromosomeFg = max(F1, F0 * self.zeroFitness / nowFitness)

    def fitting(self, isOffspring=True):
        super().fitting(isOffspring)
        if isOffspring is False:
            # if isOffspring is True:
            #     valDim = self.MIDDLE_CHROMOSOME_DIM_INDEX
            # else:
            valDim = self.CHROMOSOME_DIM_INDEX

            CR0 = self.ECArgsDictValueController["DiffCR0"]
            CR1 = self.ECArgsDictValueController["DiffCR1"]
            avgFitness = np.average(self.chromosomesFittingValue[:, valDim])
            minFitness = np.min(self.chromosomesFittingValue[:, valDim])
            for i in range(self.Np):
                if self.chromosomesFittingValue[i, valDim] > avgFitness:
                    self.chromosomeCRg[i] = CR1
                else:
                    self.chromosomeCRg[i] = CR0 * ((CR1 - CR0) * (avgFitness - self.chromosomesFittingValue[i, valDim])) / (
                                avgFitness - minFitness)