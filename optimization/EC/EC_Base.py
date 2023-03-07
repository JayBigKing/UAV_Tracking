#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : EC_Base.py
@Author  : jay.zhu
@Time    : 2022/10/8 15:00
"""

import bisect
import numpy as np
import random
from enum import Enum
from optimization.common.ArgsDictValueController import ArgsDictValueController
from optimization.common.optimizationCommonEnum import OptimizationWay


class EC_CodingType(Enum):
    BINARY_CODING = 0
    GRAY_CODING = 1
    FLOAT_CODING = 2
    SYMBOL_CODING = 4
    OTHER_CODING = 5


class EC_SelectType(Enum):
    ROULETTE = 0,
    TOUR = 1,


class EC_Base:
    '''
    evolution computation base class
    '''

    # default args for EC optimization
    DEFAULT_EC_BASE_ARGS = {
        "floatMutationOperateArg": 0.8,
        "floatCrossoverAlpha": 0.5,
        "mutationProbability": 0.05,
        "fittingMinDenominator": 0.2,
    }
    # for record parents and offspring's fitting val via one np.array,
    # use two dim to identify the two
    CHROMOSOME_DIM_INDEX = 0
    MIDDLE_CHROMOSOME_DIM_INDEX = 1

    # use two dim to identify the best chromosome over all Gen and the best chromosome in now Gen
    BEST_IN_ALL_GEN_DIM_INDEX = 0
    BEST_IN_NOW_GEN_DIM_INDEX = 1

    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes,
                 ECArgs, otherTerminalHandler=None, useCuda=False):
        self.baseInit(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes,
                      ECArgs, otherTerminalHandler, useCuda)

    '''
    initial process below
    '''

    def baseInit(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes,
                 ECArgs, otherTerminalHandler=None, useCuda=False):
        self.Np = n
        self.dimNum = dimNum
        self.maxConstraint = maxConstraint
        self.minConstraint = minConstraint
        self.evalVars = evalVars
        if isinstance(otimizeWay, OptimizationWay) is False:
            raise TypeError("otimizeWay should be an instance of enum OptimizationWay")
        else:
            self.optimizeWay = otimizeWay
        # 用最后一个位置来记录最优值
        self.chromosomesFittingValue = np.zeros((n, 2))
        self.chromosomesAimFuncValue = np.zeros((n, 2))
        self.bestChromosomesFittingValue = np.zeros((2))
        self.bestChromosomesAimFuncValue = np.zeros((2))
        self.bestChromosomeIndex = 0

        self._nowEpochTime = 0
        self.needEpochTimes = needEpochTimes
        self.ECArgsDictValueController = ArgsDictValueController(ECArgs, self.DEFAULT_EC_BASE_ARGS)
        self.ECArgs = self.ECArgsDictValueController.userArgsDict
        self.ECArgs.update(ECArgs)
        self.borders = self.ECArgs["borders"]  # 如果没有self.ECArgs["borders"] 的话，self.borders就是[]
        self.otherTerminalHandler = otherTerminalHandler
        self.useCuda = useCuda

        self.codingInit(self.Np, self.dimNum, self.ECArgs["EC_CodingType"])
        self.selectTypeInit(self.ECArgs["EC_ChoosingType"])
        self.chromosomeInit()
        self.firstRun = True

    def codingInit(self, n, dimNum, codingType: EC_CodingType = EC_CodingType.FLOAT_CODING):
        if codingType != []:
            self.EC_Base_codingType = codingType
        else:
            self.EC_Base_codingType = EC_CodingType.FLOAT_CODING

        if self.EC_Base_codingType.value == EC_CodingType.BINARY_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.GRAY_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.FLOAT_CODING.value:
            self.chromosomes = np.zeros((dimNum, n))
            self.middleChromosomes = np.zeros((dimNum, n))
            self.bestChromosome = np.zeros((dimNum, 2))
        elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
            pass

    def selectTypeInit(self, selectType: EC_SelectType = EC_SelectType.ROULETTE):
        if selectType != []:
            self.EC_Base_selectType = selectType
        else:
            self.EC_Base_selectType = EC_SelectType.ROULETTE

    def chromosomeInit(self):
        for i in range(0, self.Np):
            for j in range(0, self.dimNum):
                self.chromosomes[j, i] = self.minConstraint[j] + np.random.random() * (
                        self.maxConstraint[j] - self.minConstraint[j])
            # self.chromosomesFittingValue[i][self.CHROMOSOME_DIM_INDEX], \
            # self.chromosomesAimFuncValue[i][self.CHROMOSOME_DIM_INDEX] = \
            #     self.fittingOne(self.chromosomes[:, i], self.evalVars)
            # self.cmpToBestChromosomeAndStore(i, self.CHROMOSOME_DIM_INDEX)

        # self.fitting(isOffspring=False)

    '''
    some helpful computation below
    '''

    def limitChromosomeValue(self, chromosomeValue, dimIndex):
        if self.borders == []:
            return chromosomeValue
        elif self.borders[dimIndex] != 1:
            return chromosomeValue
        else:
            if self.EC_Base_codingType.value == EC_CodingType.BINARY_CODING.value:
                pass
            elif self.EC_Base_codingType.value == EC_CodingType.GRAY_CODING.value:
                pass
            elif self.EC_Base_codingType.value == EC_CodingType.FLOAT_CODING.value:
                return min(self.maxConstraint[dimIndex], max(chromosomeValue, self.minConstraint[dimIndex]))
            elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
                pass
            elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
                pass

    '''
    continue condition judgement and initial below
    '''

    def shouldContinue(self, otherTerminalHandler=None):
        self._nowEpochTime += 1
        if otherTerminalHandler is not None:
            if otherTerminalHandler(
                    bestChromosomesFittingValue=self.bestChromosomesFittingValue[
                        self.BEST_IN_ALL_GEN_DIM_INDEX]) is False:
                return False

        if self._nowEpochTime <= self.needEpochTimes:
            return True
        else:
            return False

    def initShouldContinueVar(self, otherTerminalHandler=None):
        if otherTerminalHandler is not None:
            otherTerminalHandler(initFlag=True)
        self._nowEpochTime = 0

    '''
    evolution computation process functions below
    '''

    def optimize(self, **kwargs):
        if self.firstRun == True:
            self.chromosomeInit()
            # self.bestChromosome = np.zeros((self.dimNum, 2))
            self.fitting(isOffspring=False)
            self.firstRun = False
        self.initShouldContinueVar(self.otherTerminalHandler)
        while self.shouldContinue(self.otherTerminalHandler):
            self.optimizeInner()

        return np.array(self.bestChromosome[:, self.BEST_IN_ALL_GEN_DIM_INDEX]), \
               self.bestChromosomesAimFuncValue[self.BEST_IN_ALL_GEN_DIM_INDEX], \
               self.bestChromosomesFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX]

    def optimizeInner(self):
        self.clearBestChromosome(self.BEST_IN_NOW_GEN_DIM_INDEX)
        self.mutation()
        self.crossover()
        self.fitting()
        self.select()

    def mutation(self):
        # mutationProbability = self.ECArgs["mutationProbability"] if self.ECArgs.get("mutationProbability") else \
        #     self.DEFAULT_EC_BASE_ARGS["mutationProbability"]
        mutationProbability = self.ECArgsDictValueController["mutationProbability"]
        for i in range(0, self.Np):
            for j in range(0, self.dimNum):
                if self.EC_Base_codingType.value == EC_CodingType.BINARY_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.GRAY_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.FLOAT_CODING.value:
                    # k = self.ECArgs["floatMutationOperateArg"] if self.ECArgs.get("floatMutationOperateArg") else \
                    #     self.DEFAULT_EC_BASE_ARGS["floatMutationOperateArg"]
                    k = self.ECArgsDictValueController["floatMutationOperateArg"]
                    if np.random.random() < mutationProbability:
                        if np.random.choice([-1., 1.]) == 1:
                            self.middleChromosomes[j][i] = self.limitChromosomeValue(
                                self.chromosomes[j][i] + k * np.random.random() * (
                                        self.maxConstraint[j] - self.chromosomes[j][i]), j)
                        else:
                            self.middleChromosomes[j][i] = self.limitChromosomeValue(
                                self.chromosomes[j][i] - k * np.random.random() * (
                                        self.chromosomes[j][i] - self.minConstraint[j]), j)
                    else:
                        self.middleChromosomes[j][i] = self.chromosomes[j][i]
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
            # alpha = self.ECArgs["crossoverAlpha"] if self.ECArgs.get("crossoverAlpha") else self.DEFAULT_EC_BASE_ARGS[
            #     "floatCrossoverAlpha"]
            alpha = self.ECArgsDictValueController["floatCrossoverAlpha"]
            alphaRemain = 1. - alpha
            for i in range(self.Np):
                r1, r2 = random.sample(range(self.Np), 2)
                for j in range(self.dimNum):
                    self.middleChromosomes[j][i] = alpha * self.middleChromosomes[j][r2] + \
                                                   alphaRemain * self.middleChromosomes[j][r1]

        elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
            pass

    def select(self):
        selectIndexSet = self.EC_Base_selectInner()
        nextChromosome = np.zeros((self.dimNum, self.Np))
        candidateAimFuncValue = np.array(self.chromosomesAimFuncValue)
        candidateFittingValue = np.array(self.chromosomesFittingValue)
        for i, selectIndex in enumerate(selectIndexSet):
            if selectIndex >= self.Np:
                nextChromosome[:, i] = self.middleChromosomes[:, selectIndex - self.Np]
                self.chromosomesAimFuncValue[i][self.CHROMOSOME_DIM_INDEX] = \
                    candidateAimFuncValue[selectIndex - self.Np][self.MIDDLE_CHROMOSOME_DIM_INDEX]
                self.chromosomesFittingValue[i][self.CHROMOSOME_DIM_INDEX] = \
                    candidateFittingValue[selectIndex - self.Np][self.MIDDLE_CHROMOSOME_DIM_INDEX]
            else:
                nextChromosome[:, i] = self.chromosomes[:, selectIndex]
                self.chromosomesAimFuncValue[i][self.CHROMOSOME_DIM_INDEX] = \
                    candidateAimFuncValue[selectIndex][self.CHROMOSOME_DIM_INDEX]
                self.chromosomesFittingValue[i][self.CHROMOSOME_DIM_INDEX] = \
                    candidateFittingValue[selectIndex][self.CHROMOSOME_DIM_INDEX]

        self.chromosomes = np.array(nextChromosome)

    def EC_Base_selectInner(self):
        selectNum = self.Np * 2  # 现在的染色体，包括本代和middle染色体，middle染色体的index为：selcctNum - self.Np
        selectIndexSet = {self.bestChromosomeIndex}
        if self.EC_Base_codingType.value == EC_CodingType.BINARY_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.GRAY_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.FLOAT_CODING.value:
            if self.EC_Base_selectType == EC_SelectType.ROULETTE:
                selectIndexSet = self.EC_Base_selectInnerForRoulette(selectNum, selectIndexSet)
            elif self.EC_Base_selectType == EC_SelectType.TOUR:
                selectIndexSet = self.EC_Base_selectInnerForTour(selectNum)
        elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
            pass

        return selectIndexSet

    def EC_Base_selectInnerForRoulette(self, selectNum, selectIndexSet):
        selectCount = 1
        totalFittingValue = np.sum(self.chromosomesFittingValue[0:self.Np, :])
        fittingProbabilityCount = 0.
        fittingProbability = [self.chromosomesFittingValue[i, self.CHROMOSOME_DIM_INDEX] / totalFittingValue for
                              i in range(self.Np)]
        fittingProbability.extend(
            [self.chromosomesFittingValue[i, self.MIDDLE_CHROMOSOME_DIM_INDEX] / totalFittingValue for i in
             range(self.Np)])
        for i in range(selectNum):
            fittingProbabilityCount += fittingProbability[i]
            fittingProbability[i] = fittingProbabilityCount

        while selectCount < self.Np:
            randomNum = np.random.random()
            selectIndex = bisect.bisect_left(fittingProbability, randomNum)
            if selectIndex not in selectIndexSet:
                selectCount += 1
                selectIndexSet.add(selectIndex)

        return selectIndexSet

    def EC_Base_selectInnerForTour(self, selectNum):
        selectIndexSet = [self.bestChromosomeIndex]
        for i in range(1, self.Np):
            parent1Idx = np.random.randint(0, selectNum)
            parent2Idx = np.random.randint(0, selectNum)
            parent1FitVal = self.chromosomesFittingValue[
                parent1Idx, self.CHROMOSOME_DIM_INDEX] if parent1Idx < self.Np else self.chromosomesFittingValue[
                parent1Idx - self.Np, self.MIDDLE_CHROMOSOME_DIM_INDEX]
            parent2FitVal = self.chromosomesFittingValue[
                parent2Idx, self.CHROMOSOME_DIM_INDEX] if parent2Idx < self.Np else self.chromosomesFittingValue[
                parent2Idx - self.Np, self.MIDDLE_CHROMOSOME_DIM_INDEX]

            if self.cmpFitting(parent1FitVal, parent2FitVal) > 0:
                selectIndexSet.append(parent1Idx)
            else:
                selectIndexSet.append(parent2Idx)

        return selectIndexSet

    '''
    fitting function and fitting process below
    '''

    def callAimFunc(self, chromosome, evalVars):
        return evalVars(chromosome)

    def fitting(self, isOffspring=True):
        if isOffspring is True:
            valDim = self.MIDDLE_CHROMOSOME_DIM_INDEX
            calcChromosomes = self.middleChromosomes
        else:
            valDim = self.CHROMOSOME_DIM_INDEX
            calcChromosomes = self.chromosomes

        for i in range(self.Np):
            self.chromosomesFittingValue[i][valDim], self.chromosomesAimFuncValue[i][
                valDim] = self.fittingOne(
                calcChromosomes[:, i], self.evalVars)
            self.cmpToBestChromosomeAndStore(i, valDim)

    def fittingOne(self, chromosome, evalVars):
        aimFuncVal = self.callAimFunc(chromosome, evalVars)
        fittingValue = aimFuncVal
        if self.optimizeWay == OptimizationWay.MIN:
            # fittingMinDenominator = self.ECArgs["fittingMinDenominator"] if self.ECArgs.get(
            #     "fittingMinDenominator") else self.DEFAULT_EC_BASE_ARGS["fittingMinDenominator"]
            fittingMinDenominator = self.ECArgsDictValueController["fittingMinDenominator"]
            fittingValue = 1 / (fittingValue + fittingMinDenominator)
        return fittingValue, aimFuncVal

    def cmpFitting(self, val1, val2):
        return (val1 - val2)

    def clearBestChromosome(self, whichBestChromosome):
        if whichBestChromosome == self.BEST_IN_NOW_GEN_DIM_INDEX:
            self.bestChromosomesFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX], self.bestChromosomesAimFuncValue[
                self.BEST_IN_NOW_GEN_DIM_INDEX] = 0., 0.
        elif whichBestChromosome == self.BEST_IN_ALL_GEN_DIM_INDEX:
            self.bestChromosomesFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX], self.bestChromosomesAimFuncValue[
                self.BEST_IN_ALL_GEN_DIM_INDEX] = 0., 0.

    def cmpToBestChromosomeAndStore(self, index, valDim):
        if self.cmpFitting(self.chromosomesFittingValue[index][valDim],
                           self.bestChromosomesFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX]) > 0:

            self.bestChromosomesFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX], self.bestChromosomesAimFuncValue[
                self.BEST_IN_NOW_GEN_DIM_INDEX] = \
                self.chromosomesFittingValue[index][valDim], self.chromosomesAimFuncValue[index][valDim]
            if valDim == self.CHROMOSOME_DIM_INDEX:
                self.bestChromosomeIndex = index
                self.bestChromosome[:, self.BEST_IN_NOW_GEN_DIM_INDEX] = np.array(self.chromosomes[:, index])
            else:
                self.bestChromosomeIndex = index + self.Np
                self.bestChromosome[:, self.BEST_IN_NOW_GEN_DIM_INDEX] = np.array(self.middleChromosomes[:, index])

            if self.cmpFitting(self.bestChromosomesFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX],
                               self.bestChromosomesFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX]) > 0:
                self.bestChromosomesFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX], self.bestChromosomesAimFuncValue[
                    self.BEST_IN_ALL_GEN_DIM_INDEX] = self.bestChromosomesFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX], \
                                                      self.bestChromosomesAimFuncValue[self.BEST_IN_NOW_GEN_DIM_INDEX]
                self.bestChromosome[:, self.BEST_IN_ALL_GEN_DIM_INDEX] = np.array(
                    self.bestChromosome[:, self.BEST_IN_NOW_GEN_DIM_INDEX])
