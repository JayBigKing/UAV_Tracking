#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : EC_DiffEC_Base.py
@Author  : jay.zhu
@Time    : 2022/12/15 0:19
"""
import numpy as np
import random
from Jay_Tool.LogTool import myLogger
from optimization.EC.EC_Base import EC_CodingType
from optimization.EC.EC_WithStat_Base import EC_WithStat_Base


class EC_DiffEC_Base(EC_WithStat_Base):
    # default args for diff EC optimization
    __DIFF_EC_BASE_DEFAULT_ARGS = {
        "DiffCR0": 0.1,
        "DiffCR1": 0.6,
        "DiffF0": 0.1,
        "DiffF1": 0.9,
    }

    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, otherTerminalHandler=None, useCuda=False):
        try:
            if n < 3:
                raise ValueError("The num of chromosome is too small, "
                                 "Please input the num which is bigger than or equal to 3")
            else:
                super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                                 statRegisters, otherTerminalHandler, useCuda)

                self.ECArgsDictValueController.update(newDict=self.__DIFF_EC_BASE_DEFAULT_ARGS, onlyAddNotExists=True)
                self.chromosomeCRg = np.zeros(self.Np)

        except ValueError as e:
            myLogger.myLogger_Logger().error(repr(e))

    def mutation(self):
        F0 = self.ECArgsDictValueController["DiffF0"]
        F1 = self.ECArgsDictValueController["DiffF1"]
        mutationProbability = self.ECArgsDictValueController["mutationProbability"]
        for i in range(0, self.Np):
            r1, r2, r3 = random.sample(range(self.Np), 3)
            rFitnessList = [self.chromosomesFittingValue[r1, self.CHROMOSOME_DIM_INDEX],
                            self.chromosomesFittingValue[r2, self.CHROMOSOME_DIM_INDEX],
                            self.chromosomesFittingValue[r3, self.CHROMOSOME_DIM_INDEX]]
            rIndexList = [r1, r2, r3]
            bestRIndex = r1
            worstRIndex = r1
            for i in range(len(rFitnessList)):
                if self.cmpFitting(rFitnessList[i],
                                   self.chromosomesFittingValue[bestRIndex, self.CHROMOSOME_DIM_INDEX]) > 0:
                    bestRIndex = rIndexList[i]
                if self.cmpFitting(rFitnessList[i],
                                   self.chromosomesFittingValue[worstRIndex, self.CHROMOSOME_DIM_INDEX]) < 0:
                    worstRIndex = rIndexList[i]
            if bestRIndex == worstRIndex:
                bestRIndex, middleRIndex, worstRIndex = r1, r2, r3
            else:
                middleRIndex = ({r1, r2, r3} - {bestRIndex, worstRIndex}).pop()

            if self.chromosomesFittingValue[middleRIndex, self.CHROMOSOME_DIM_INDEX] == self.chromosomesFittingValue[
                worstRIndex, self.CHROMOSOME_DIM_INDEX]:
                Fg = (F1 - F0) / 2.
            elif self.chromosomesFittingValue[middleRIndex, self.CHROMOSOME_DIM_INDEX] == self.chromosomesFittingValue[
                bestRIndex, self.CHROMOSOME_DIM_INDEX]:
                Fg = (F1 - F0) / 2.
            else:
                Fg = F0 + (F1 - F0) * (self.chromosomesFittingValue[middleRIndex, self.CHROMOSOME_DIM_INDEX] -
                                       self.chromosomesFittingValue[worstRIndex, self.CHROMOSOME_DIM_INDEX]) / (
                             self.chromosomesFittingValue[bestRIndex, self.CHROMOSOME_DIM_INDEX] -
                             self.chromosomesFittingValue[middleRIndex, self.CHROMOSOME_DIM_INDEX])
            for j in range(0, self.dimNum):
                if self.EC_Base_codingType.value == EC_CodingType.BINARY_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.GRAY_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.FLOAT_CODING.value:
                    # if np.random.random() < mutationProbability:
                    self.middleChromosomes[j, i] = min(
                        max(self.chromosomes[j, worstRIndex] + Fg * (
                                self.chromosomes[j, bestRIndex] - self.chromosomes[j, middleRIndex]),
                            self.minConstraint[j]),
                        self.maxConstraint[j])
                # else:
                #     self.middleChromosomes[j, i] = self.chromosomes[j, i]
                elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
                    pass

    def crossover(self):
        if self.EC_Base_codingType.value == EC_CodingType.BINARY_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.GRAY_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.FLOAT_CODING.value:
            for i in range(self.Np):
                for j in range(self.dimNum):
                    if random.random() > self.chromosomeCRg[i]:
                        self.middleChromosomes[j][i] = self.chromosomes[j][i]

        elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
            pass

    def EC_Base_selectInner(self):
        selectIndexSet = []

        if self.EC_Base_codingType.value == EC_CodingType.BINARY_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.GRAY_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.FLOAT_CODING.value:
            for i in range(self.Np):
                if self.cmpFitting(self.chromosomesFittingValue[i, self.CHROMOSOME_DIM_INDEX],
                                   self.chromosomesFittingValue[i, self.MIDDLE_CHROMOSOME_DIM_INDEX]) > 0:
                    selectIndexSet.append(i)
                else:
                    selectIndexSet.append(i + self.Np)
        elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
            pass

        return selectIndexSet

    def updateDiffArg(self):
        CR0 = self.ECArgsDictValueController["DiffCR0"]
        CR1 = self.ECArgsDictValueController["DiffCR1"]
        valDim = self.CHROMOSOME_DIM_INDEX
        for i in range(self.Np):
            if self.cmpFitting(self.chromosomesFittingValue[i, valDim], self.avgFitness) < 0:
                self.chromosomeCRg[i] = CR1
            else:
                self.chromosomeCRg[i] = CR0 * (
                        (CR1 - CR0) * (self.maxFitness - self.avgFitness) / (
                self.chromosomesFittingValue[i, valDim]) - self.avgFitness)

    def fitting(self, isOffspring=True):
        super().fitting(isOffspring)
        if isOffspring is False:
            valDim = self.CHROMOSOME_DIM_INDEX

            self.avgFitness = np.average(self.chromosomesFittingValue[:, valDim])
            self.maxFitness = np.min(self.chromosomesFittingValue[:, valDim])

            self.updateDiffArg()
