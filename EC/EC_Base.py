import collections
import bisect
import numpy as np
import random
from enum import Enum
from abc import ABC, abstractmethod


class EC_CodingType(Enum):
    BINARY_CODING = 0
    GRAY_CODING = 1
    FLOAT_CODING = 2
    SYMBOL_CODING = 4
    OTHER_CODING = 5


class EC_SelectType(Enum):
    ROULETTE = 0,
    TOUR = 1,


class EC_OtimizeWay(Enum):
    MAX = 1
    MIN = -1


class EC_Base:
    '''
    evolution compution base class
    '''

    # default args for EC optimization
    DEFAULT_EC_BASE_ARGS = {
        "floatMutationOperateArg": 0.8,
        "floatCrossoverAlpha": 0.5,
        "mutationProbability": 0.05,
        "fittingMinDenominator": 1.,
    }
    # for record parents and offspring's fitting val via one np.array,
    # use two dim to identify the two
    CHROMOSOME_DIM_INDEX = 0
    MIDDLE_CHROMOSOME_DIM_INDEX = 1

    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes,
                 ECArgs, otherTerminalHandler=None, useCuda=False):
        self.baseInit(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes,
                      ECArgs, otherTerminalHandler, useCuda)

    def baseInit(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes,
                 ECArgs, otherTerminalHandler=None, useCuda=False):
        self.Np = n
        self.dimNum = dimNum
        self.maxConstraint = maxConstraint
        self.minConstraint = minConstraint
        self.evalVars = evalVars
        if isinstance(otimizeWay, EC_OtimizeWay) is False:
            raise TypeError("otimizeWay should be an instance of enum EC_OtimizeWay")
        else:
            self.optimizeWay = otimizeWay
        #用最后一个位置来记录最优值
        self.chromosomesFittingValue = np.zeros((n+1, 2))
        self.chromosomesAimFuncValue = np.zeros((n+1, 2))
        self.bestChromosomeIndex = 0
        self._nowEpochTime = 0
        self.needEpochTimes = needEpochTimes
        self.ECArgs = collections.defaultdict(list)
        self.ECArgs.update(ECArgs)
        self.borders = self.ECArgs["borders"]  # 如果没有self.ECArgs["borders"] 的话，self.borders就是[]
        self.otherTerminalHandler = otherTerminalHandler
        self.useCuda = useCuda

        self.codingInit(self.Np, self.dimNum, self.ECArgs["EC_CodingType"])
        self.selectTypeInit(self.ECArgs["EC_ChoosingType"])
        self.chromosomeInit()

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
            self.bestChromosome = np.zeros(dimNum)
        elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
            pass

    def selectTypeInit(self, selectType: EC_SelectType = EC_SelectType.ROULETTE):
        if selectType != []:
            self.EC_Base_selectType = EC_SelectType
        else:
            self.EC_Base_selectType = EC_SelectType.ROULETTE

    def cmpToBestChromosomeAndStore(self, index, valDim):
        if self.cmpFitting(self.chromosomesFittingValue[index][valDim],
                           self.chromosomesFittingValue[-1][self.CHROMOSOME_DIM_INDEX]) > 0:

            self.chromosomesFittingValue[-1][self.CHROMOSOME_DIM_INDEX], self.chromosomesAimFuncValue[-1][self.CHROMOSOME_DIM_INDEX] = \
                self.chromosomesFittingValue[index][valDim], self.chromosomesAimFuncValue[index][valDim]
            if valDim == self.CHROMOSOME_DIM_INDEX:
                self.bestChromosomeIndex = index
                self.bestChromosome = np.array(self.chromosomes[:, index])
            else:
                # self.chromosomesFittingValue[index][self.CHROMOSOME_DIM_INDEX], self.chromosomesAimFuncValue[index][self.CHROMOSOME_DIM_INDEX] = \
                # self.chromosomesFittingValue[index][valDim], self.chromosomesAimFuncValue[index][valDim]
                self.bestChromosomeIndex = index + self.Np
                self.bestChromosome = np.array(self.middleChromosomes[:, index])
                # self.chromosomes[:, index] = np.array(self.middleChromosomes[:, index])

    def chromosomeInit(self):
        for i in range(0, self.Np):
            for j in range(0, self.dimNum):
                self.chromosomes[j, i] = self.minConstraint[j] + np.random.random() * (
                        self.maxConstraint[j] - self.minConstraint[j])
            self.chromosomesFittingValue[i][self.CHROMOSOME_DIM_INDEX], \
            self.chromosomesAimFuncValue[i][self.CHROMOSOME_DIM_INDEX] = \
                self.fitting(self.chromosomes[:, i], self.evalVars)
            # if self.cmpFitting(self.chromosomesFittingValue[i][self.CHROMOSOME_DIM_INDEX],
            #                    self.chromosomesFittingValue[self.bestChromosomeIndex][self.CHROMOSOME_DIM_INDEX]) > 0:
            #     self.bestChromosomeIndex = i
            self.cmpToBestChromosomeAndStore(i, self.CHROMOSOME_DIM_INDEX)

    def callAimFunc(self, chromosome, evalVars):
        return evalVars(chromosome)

    def fitting(self, chromosome, evalVars):
        aimFuncVal = self.callAimFunc(chromosome, evalVars)
        fittingValue = aimFuncVal
        if self.optimizeWay == EC_OtimizeWay.MIN:
            fittingMinDenominator = self.ECArgs["fittingMinDenominator"] if self.ECArgs.get(
                "fittingMinDenominator") else self.DEFAULT_EC_BASE_ARGS["fittingMinDenominator"]
            fittingValue = 1 / (fittingValue + 1 + fittingMinDenominator)
        return fittingValue, aimFuncVal

    def cmpFitting(self, val1, val2):
        return (val1 - val2)

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

    def optimize(self):
        while self.shouldContinue(self.otherTerminalHandler):
            self.optimizeInner()

        return np.array(self.bestChromosome), \
               self.chromosomesAimFuncValue[-1][self.CHROMOSOME_DIM_INDEX]

    def optimizeInner(self):
        self.mutation()
        self.crossover()
        self.select()

    def shouldContinue(self, otherTerminalHandler=None):
        self._nowEpochTime += 1
        if otherTerminalHandler is not None:
            if otherTerminalHandler(
                    self.chromosomesFittingValue[-1, self.CHROMOSOME_DIM_INDEX]) is True:
                return True
        if self._nowEpochTime <= self.needEpochTimes:
            return True
        else:
            return False

    def mutation(self):
        mutationProbability = self.ECArgs["mutationProbability"] if self.ECArgs.get("mutationProbability") else \
            self.DEFAULT_EC_BASE_ARGS["mutationProbability"]
        for i in range(0, self.Np):
            for j in range(0, self.dimNum):
                if self.EC_Base_codingType.value == EC_CodingType.BINARY_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.GRAY_CODING.value:
                    pass
                elif self.EC_Base_codingType.value == EC_CodingType.FLOAT_CODING.value:
                    k = self.ECArgs["floatMutationOperateArg"] if self.ECArgs.get("floatMutationOperateArg") else \
                        self.DEFAULT_EC_BASE_ARGS["floatMutationOperateArg"]
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
            alpha = self.ECArgs["crossoverAlpha"] if self.ECArgs.get("crossoverAlpha") else self.DEFAULT_EC_BASE_ARGS[
                "floatCrossoverAlpha"]
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

        for i in range(self.Np):
            self.chromosomesFittingValue[i][self.MIDDLE_CHROMOSOME_DIM_INDEX], self.chromosomesAimFuncValue[i][
                self.MIDDLE_CHROMOSOME_DIM_INDEX] = self.fitting(
                self.middleChromosomes[:, i], self.evalVars)
            # if self.cmpFitting(self.chromosomesFittingValue[i][self.MIDDLE_CHROMOSOME_DIM_INDEX],
            #                    self.chromosomesFittingValue[self.bestChromosomeIndex][self.CHROMOSOME_DIM_INDEX]) > 0:
            #     self.bestChromosomeIndex = i
            #     self.chromosomesFittingValue[i][self.CHROMOSOME_DIM_INDEX], self.chromosomesAimFuncValue[i][
            #         self.CHROMOSOME_DIM_INDEX] = \
            #         self.chromosomesFittingValue[i][self.MIDDLE_CHROMOSOME_DIM_INDEX], self.chromosomesAimFuncValue[i][
            #             self.MIDDLE_CHROMOSOME_DIM_INDEX]
            #     self.chromosomes[:, i] = np.array(self.middleChromosomes[:, i])
            self.cmpToBestChromosomeAndStore(i, self.MIDDLE_CHROMOSOME_DIM_INDEX)

    def select(self):
        selectIndexSet = self.EC_Base_selectInner()
        nextChromosome = np.zeros((self.dimNum, self.Np))
        for i, selectIndex in enumerate(selectIndexSet):
            if selectIndex >= self.Np:
                nextChromosome[:, i] = self.middleChromosomes[:, selectIndex - self.Np]
                self.chromosomesAimFuncValue[i][self.CHROMOSOME_DIM_INDEX] = \
                    self.chromosomesAimFuncValue[selectIndex - self.Np][self.MIDDLE_CHROMOSOME_DIM_INDEX]
                self.chromosomesFittingValue[i][self.CHROMOSOME_DIM_INDEX] = \
                    self.chromosomesFittingValue[selectIndex - self.Np][self.MIDDLE_CHROMOSOME_DIM_INDEX]
            else:
                nextChromosome[:, i] = self.chromosomes[:, selectIndex]

        # nextChromosome[:, self.bestChromosomeIndex] = self.chromosomes[:, self.bestChromosomeIndex]
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
            elif self.EC_Base_selectType == EC_SelectType.TOUR:
                pass
        elif self.EC_Base_codingType.value == EC_CodingType.SYMBOL_CODING.value:
            pass
        elif self.EC_Base_codingType.value == EC_CodingType.OTHER_CODING.value:
            pass

        return selectIndexSet
