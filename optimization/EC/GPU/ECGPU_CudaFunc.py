#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : ECGPU_CudaFunc.py
@Author  : jay.zhu
@Time    : 2023/2/26 20:19
"""
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

@cuda.jit(device=True)
def gpu_randint(rngStates, threadIdx, a, b):
    randRes = xoroshiro128p_uniform_float32(rngStates, threadIdx)
    return int(a + randRes * (b - a + 1))


@cuda.jit(device=True)
def ECGPU_LimitChromosomeValue(chromosomeValue, dimIndex, maxConstraint, minConstraint):
    return min(maxConstraint[dimIndex], max(chromosomeValue, minConstraint[dimIndex]))


@cuda.jit(device=True)
def ECGPU_CmpFitting(val1, val2):
    return (val1 - val2)


@cuda.jit
def ECGPU_InitChromosomes(chromosomes, maxConstraint, minConstraint, rngStates):
    # 通过GPU并行化初始化种群
    i, j = cuda.grid(2)
    if i < chromosomes.shape[0] and j < chromosomes.shape[1]:
        chromosomes[i][j] = minConstraint[i] + xoroshiro128p_uniform_float32(rngStates, i * cuda.gridDim.x + j) * (
                maxConstraint[i] - minConstraint[i])


@cuda.jit
def ECGPU_Mutation(chromosomes, middleChromosomes, maxConstraint, minConstraint, mutationProbability,
                   floatMutationOperateArg, rngStates):
    # 通过GPU并行化初始化种群
    i, j = cuda.grid(2)
    k = floatMutationOperateArg
    if i < chromosomes.shape[0] and j < chromosomes.shape[1]:
        if xoroshiro128p_uniform_float32(rngStates, i * cuda.gridDim.x + j) < mutationProbability:
            if gpu_randint(rngStates, i * cuda.gridDim.x + j, 0, 1) == 1:
                middleChromosomes[i][j] = ECGPU_LimitChromosomeValue(
                    chromosomes[i][j] + k * xoroshiro128p_uniform_float32(rngStates, i * cuda.gridDim.x + j) * (
                            maxConstraint[i] - chromosomes[i][j]), i, maxConstraint, minConstraint)
            else:
                middleChromosomes[i][j] = ECGPU_LimitChromosomeValue(
                    chromosomes[i][j] - k * xoroshiro128p_uniform_float32(rngStates, i * cuda.gridDim.x + j) * (
                            chromosomes[i][j] - minConstraint[i]), i, maxConstraint, minConstraint)
        else:
            middleChromosomes[i][j] = chromosomes[i][j]


# 交叉操作
@cuda.jit
def ECGPU_Crossover(offspring, middleChromosomes, crossoverRate, rngStates):
    # 通过GPU并行化执行交叉操作
    idx = cuda.grid(1)
    if idx < offspring.shape[1]:
        if xoroshiro128p_uniform_float32(rngStates, idx) < crossoverRate:
            parent1Idx = gpu_randint(rngStates, idx, 0, offspring.shape[1])
            parent2Idx = gpu_randint(rngStates, idx, 0, offspring.shape[1])
            crossoverPos = gpu_randint(rngStates, idx, 0, offspring.shape[0])
            for i in range(0, crossoverPos):
                offspring[i][idx] = middleChromosomes[i][parent1Idx]
            for i in range(crossoverPos, offspring.shape[0]):
                offspring[i][idx] = middleChromosomes[i][parent2Idx]
        else:
            for i in range(0, offspring.shape[0]):
                offspring[i, idx] = middleChromosomes[i, idx]


@cuda.jit(device=True)
def ECGPU_AimFunc(chromosomes, idx, dataForAimFunc):
    pass


@cuda.jit(device=True)
def ECGPU_FittingOne(chromosomes, idx, dataForAimFunc, fittingMinDenominator):
    aimFuncVal = ECGPU_AimFunc(chromosomes, idx, dataForAimFunc)
    fittingValue = aimFuncVal

    if fittingMinDenominator > 0:
        fittingValue = 1 / (fittingValue + fittingMinDenominator)
    return fittingValue, aimFuncVal


# 计算适应度函数
@cuda.jit
def ECGPU_Fitness(chromosomes, chromosomesFittingValue, chromosomesAimFuncValue, dataForAimFunc, fittingMinDenominator,
                  valDim):
    # 通过GPU并行化计算适应度函数
    idx = cuda.grid(1)

    if idx < chromosomes.shape[1]:
        chromosomesFittingValue[idx, valDim], chromosomesAimFuncValue[idx, valDim] = ECGPU_FittingOne(chromosomes, idx,
                                                                                                      dataForAimFunc,
                                                                                                      fittingMinDenominator)


@cuda.jit
def ECGPU_CmpToBestChromosomeAndStore(chromosomes, bestChromosome, chromosomesFittingValue, chromosomesAimFuncValue,
                                      bestChromosomesFittingValue, bestChromosomesAimFuncValue, valDim,
                                      bestInNowGenDimIndex, bestInAllGenDimIndex):
    # 通过GPU并行化计算适应度函数
    idx = cuda.grid(1)

    if idx == 0:
        for i in range(chromosomes.shape[1]):
            if ECGPU_CmpFitting(chromosomesFittingValue[i, valDim], bestChromosomesFittingValue[bestInNowGenDimIndex]) > 0:
                bestChromosomesFittingValue[bestInNowGenDimIndex], bestChromosomesAimFuncValue[bestInNowGenDimIndex] = \
                    chromosomesFittingValue[i, valDim], chromosomesAimFuncValue[i, valDim]

                for j in range(chromosomes.shape[0]):
                    bestChromosome[j, bestInNowGenDimIndex] = chromosomes[j, i]

                if ECGPU_CmpFitting(bestChromosomesFittingValue[bestInNowGenDimIndex],
                                    bestChromosomesFittingValue[bestInAllGenDimIndex]) > 0:
                    bestChromosomesFittingValue[bestInAllGenDimIndex], bestChromosomesAimFuncValue[bestInAllGenDimIndex] = \
                        bestChromosomesFittingValue[bestInNowGenDimIndex], bestChromosomesAimFuncValue[bestInNowGenDimIndex]

                    for j in range(chromosomes.shape[0]):
                        bestChromosome[j, bestInAllGenDimIndex] = bestChromosome[j, bestInNowGenDimIndex]


@cuda.jit
def ECGPU_SelectionPrepareCandidates(candidates, chromosomes, midOffspring):
    i, j = cuda.grid(2)
    if i < chromosomes.shape[0] and j < chromosomes.shape[1] + chromosomes.shape[1]:
        if j < chromosomes.shape[1]:
            candidates[i, j] = chromosomes[i, j]
        else:
            candidates[i, j] = midOffspring[i, j - chromosomes.shape[1]]


@cuda.jit
def ECGPU_SelectionPrepareFitness(candidatesFittingValue, candidatesAimFuncValue, chromosomesFittingValue,
                                  chromosomesAimFuncValue, chromosomeDimIndex, middleChromosomeDimIndex, ):
    idx = cuda.grid(1)
    if idx < candidatesFittingValue.shape[0]:
        if idx < chromosomesFittingValue.shape[0]:
            candidatesFittingValue[idx], candidatesAimFuncValue[idx] = chromosomesFittingValue[idx][chromosomeDimIndex], \
                                                                       chromosomesAimFuncValue[idx][chromosomeDimIndex]
        else:
            candidatesFittingValue[idx], candidatesAimFuncValue[idx] = \
                chromosomesFittingValue[idx - chromosomesFittingValue.shape[0]][middleChromosomeDimIndex], \
                chromosomesAimFuncValue[idx - chromosomesFittingValue.shape[0]][middleChromosomeDimIndex]


@cuda.jit
def ECGPU_SelectionPrepareFittingProbForRoulette(candidatesFittingValue, fittingProb, candidateSelectCheckList):
    idx = cuda.grid(1)
    if idx == 0:
        totalFittingValue = 0.
        fittingProbabilityCount = 0.
        for i in range(candidatesFittingValue.shape[0]):
            totalFittingValue += candidatesFittingValue[i]

        for i in range(fittingProb.shape[0]):
            fittingProb[i] = fittingProbabilityCount
            fittingProbabilityCount += (candidatesFittingValue[i] / totalFittingValue)
            candidateSelectCheckList[i] = 0


@cuda.jit
def ECGPU_SelectOneChromosomeForRoulette(candidates, fittingProb, candidateSelectCheckList, rngStates):
    idx = cuda.grid(1)
    if idx < candidates.shape[1]:
        for i in range(10):
            if candidateSelectCheckList[idx] != 1:
                randVal = xoroshiro128p_uniform_float32(rngStates, idx)
                if fittingProb[idx] < randVal < fittingProb[(idx + 1)]:
                    candidateSelectCheckList[idx] = 1


@cuda.jit
def ECGPU_SelectCheckDoneForRoulette(chromosomesNum, candidateSelectCheckList,
                          candidateSelectIndexList, selectDoneFlag,
                          rngStates):
    idx = cuda.grid(1)
    if idx == 0:
        selectDoneFlag[0] = 0
        selectNum = 0
        for i in range(candidateSelectCheckList.shape[0]):
            selectNum += candidateSelectCheckList[i]
        if selectNum < chromosomesNum:
            return
        else:
            selectDoneFlag[0] = 1
            selectNum = 0
            if xoroshiro128p_uniform_float32(rngStates, idx) < 0.5:
                for i in range(candidateSelectCheckList.shape[0], 0, -1):
                    if candidateSelectCheckList[i] == 1:
                        candidateSelectIndexList[selectNum] = i
                        selectNum += 1
                    if selectNum == chromosomesNum:
                        break
            else:
                for i in range(0, candidateSelectCheckList.shape[0]):
                    if candidateSelectCheckList[i] == 1:
                        candidateSelectIndexList[selectNum] = i
                        selectNum += 1
                    if selectNum == chromosomesNum:
                        break


@cuda.jit
def ECGPU_SelectOffspringForRoulette(chromosomes, candidates, candidateSelectIndexList, candidatesFittingValue,
                          candidatesAimFuncValue,
                          chromosomesFittingValue, chromosomesAimFuncValue, chromosomeDimIndex):
    idx = cuda.grid(1)
    if idx < chromosomes.shape[1]:
        getCandidatesIndex = int(candidateSelectIndexList[idx])
        chromosomesFittingValue[idx, chromosomeDimIndex], chromosomesAimFuncValue[idx, chromosomeDimIndex] = \
            candidatesFittingValue[getCandidatesIndex], candidatesAimFuncValue[getCandidatesIndex]
        for j in range(chromosomes.shape[0]):
            chromosomes[j, idx] = candidates[j, getCandidatesIndex]

@cuda.jit
def selectionSaveBestChromosome(chromosomes, bestChromosome, chromosomesFittingValue, chromosomesAimFuncValue,
                                bestChromosomesFittingValue, bestChromosomesAimFuncValue, bestInNowGenDimIndex,
                                chromosomeDimIndex):
    idx = cuda.grid(1)
    if idx == 0:
        for j in range(chromosomes.shape[0]):
            chromosomes[j, 0] = bestChromosome[j, bestInNowGenDimIndex]

        chromosomesFittingValue[0, chromosomeDimIndex], chromosomesAimFuncValue[0, chromosomeDimIndex] = \
            bestChromosomesFittingValue[bestInNowGenDimIndex], bestChromosomesAimFuncValue[bestInNowGenDimIndex]

@cuda.jit
def selectionForTour(chromosomes, candidates, candidatesFittingValue,candidatesAimFuncValue,
                     chromosomesFittingValue, chromosomesAimFuncValue, chromosomeDimIndex,
                     rngStates):
    # 通过GPU并行化执行选择操作
    idx = cuda.grid(1)
    if 0 < idx < chromosomes.shape[1]:
        parent1Idx = gpu_randint(rngStates, idx, 0, candidates.shape[0])
        parent2Idx = gpu_randint(rngStates, idx, 0, candidates.shape[0])
        if ECGPU_CmpFitting(candidatesFittingValue[parent1Idx], candidatesFittingValue[parent2Idx]) > 0:
            selectIndex = parent1Idx
        else:
            selectIndex = parent2Idx

        for j in range(chromosomes.shape[0]):
            chromosomes[j, idx] = candidates[j, selectIndex]

        chromosomesFittingValue[idx, chromosomeDimIndex], chromosomesAimFuncValue[idx, chromosomeDimIndex] = \
            candidatesFittingValue[selectIndex], candidatesAimFuncValue[selectIndex]


@cuda.jit
def ECGPU_ClearBestChromosome(chromosomesFittingValue, chromosomesAimFuncValue, bestChromosomeIndex):
    idx = cuda.grid(1)
    if idx == 0:
        chromosomesFittingValue[bestChromosomeIndex], chromosomesAimFuncValue[bestChromosomeIndex] = 0., 0.
