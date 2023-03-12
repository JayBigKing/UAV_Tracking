#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : ECGPU_Base.py
@Author  : jay.zhu
@Time    : 2023/2/24 20:45
"""
import time
import numpy as np
from optimization.EC.EC_Base import EC_SelectType
from optimization.EC.EC_WithStat_Base import EC_WithStat_Base
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import optimization.EC.GPU.ECGPU_CudaFunc as cudaFunc

class ECGPU_Base(EC_WithStat_Base):
    DEFAULT_EC_GPU_BASE_ARGS = {
        "threadsPerBlock1D": 5,
        "threadsPerBlock2D": [5, 1],
        "crossoverRate": 0.95,
    }
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, otherTerminalHandler=None, useCuda=True):
        super().__init__(n=n,
                         dimNum=dimNum,
                         maxConstraint=maxConstraint,
                         minConstraint=minConstraint,
                         evalVars=evalVars,
                         otimizeWay=otimizeWay,
                         needEpochTimes=needEpochTimes,
                         ECArgs=ECArgs,
                         statRegisters=statRegisters,
                         otherTerminalHandler=otherTerminalHandler,
                         useCuda=useCuda)

    def baseInit(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes,
                 ECArgs, otherTerminalHandler=None, useCuda=True):
        super().baseInit(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                      otherTerminalHandler, False)
        self.useCuda = useCuda

        self.ECArgsDictValueController.update(newDict=self.DEFAULT_EC_GPU_BASE_ARGS, onlyAddNotExists=True)

        self.ECGPU_MaxConstraint = cuda.to_device(np.array(self.maxConstraint))
        self.ECGPU_MinConstraint = cuda.to_device(np.array(self.minConstraint))

        self.ECGPU_ChromosomesFittingValue = cuda.to_device(self.chromosomesFittingValue)
        self.ECGPU_ChromosomesAimFuncValue = cuda.to_device(self.chromosomesAimFuncValue)
        self.ECGPU_BestChromosomesFittingValue = cuda.to_device(self.bestChromosomesFittingValue)
        self.ECGPU_BestChromosomesAimFuncValue = cuda.to_device(self.bestChromosomesAimFuncValue)
        self.ECGPU_BestChromosomeIndex = cuda.device_array(1)
        self.ECGPU_CandidatesFittingValue = cuda.device_array(n * 2)
        self.ECGPU_CandidatesAimFuncValue = cuda.device_array(n * 2)

        self.ECGPU_Chromosomes = cuda.to_device(self.chromosomes)
        self.ECGPU_MiddleChromosomes = cuda.to_device(self.middleChromosomes)
        self.ECGPU_BestChromosome = cuda.to_device(self.bestChromosome)
        self.ECGPU_MidOffspring = cuda.device_array((dimNum, n))
        self.ECGPU_Candidates = cuda.device_array((dimNum, n * 2))

        self.ECGPU_FittingProb = cuda.device_array(n * 2)
        self.ECGPU_CandidateSelectCheckList = cuda.device_array(n * 2)
        self.ECGPU_SelectDoneFlag = cuda.device_array(1)
        self.ECGPU_CandidateSelectIndexList = cuda.device_array(n)

        self.ECGPU_RngStates1D = create_xoroshiro128p_states(n, seed=int(time.time() * 1e7))
        self.ECGPU_RngStates2D = create_xoroshiro128p_states(n * dimNum, seed=int(time.time() * 1e7))
        self.ECGPU_RngStates1DSel = create_xoroshiro128p_states(n * 2, seed=int(time.time() * 1e7))

        self.ECGPU_DataForAimFunc = cuda.device_array(self.ECArgsDictValueController["dataForAimFuncSize"])

        self.ECGPU_ThreadsPerBlock1D, self.ECGPU_BlocksPerGrid1D = self.getBlocksPerGrid(
            self.Np, self.ECArgsDictValueController["threadsPerBlock1D"], dim=1)
        self.ECGPU_ThreadsPerBlock2D, self.ECGPU_BlocksPerGrid2D = self.getBlocksPerGrid(
            self.Np, self.ECArgsDictValueController["threadsPerBlock2D"], dim=2)
        self.ECGPU_ThreadsPerBlock1DSel, self.ECGPU_BlocksPerGrid1DSel = self.getBlocksPerGrid(
            self.Np + self.Np, self.ECArgsDictValueController["threadsPerBlock1D"], dim=1)
        self.ECGPU_ThreadsPerBlock2DSel, self.ECGPU_BlocksPerGrid2DSel = self.getBlocksPerGrid(
            self.Np + self.Np, self.ECArgsDictValueController["threadsPerBlock2D"], dim=2)


        self.updateAimFunc()

        self.chromosomeInit()

    def getBlocksPerGrid(self, chromosomeNum, threadsPerBlock, dim=1, dimNum=None):
        if dim == 1:
            if threadsPerBlock > chromosomeNum:
                threadsPerBlock = chromosomeNum
            while chromosomeNum % threadsPerBlock != 0:
                threadsPerBlock -= 1
                if threadsPerBlock <= 0:
                    raise ValueError("Np must be great than 0")
            return threadsPerBlock, int(chromosomeNum / threadsPerBlock)
        else:
            if dimNum is None:
                dimNum = self.dimNum

            sizeOfGridDim = [dimNum, chromosomeNum, ]

            for i in range(2):
                if threadsPerBlock[i] > sizeOfGridDim[i]:
                    threadsPerBlock[i] = sizeOfGridDim[i]

                while sizeOfGridDim[i] % threadsPerBlock[i] != 0:
                    threadsPerBlock[i] -= 1
                    if threadsPerBlock[i] <= 0:
                        raise ValueError("Np must be great than 0")

            return threadsPerBlock, [int(sizeOfGridDim[index] / item) for index, item in enumerate(threadsPerBlock)]

    def updateAimFunc(self):
        cudaFunc.ECGPU_AimFunc = self.evalVars

    def chromosomeInit(self):
        if self.useCuda is False:
            return
        else:
            cudaFunc.ECGPU_InitChromosomes[self.ECGPU_BlocksPerGrid2D, self.ECGPU_ThreadsPerBlock2D](self.ECGPU_Chromosomes,
                                                                                            self.ECGPU_MaxConstraint,
                                                                                            self.ECGPU_MinConstraint,
                                                                                            self.ECGPU_RngStates2D)
            cuda.synchronize()

    def mutation(self):
        cudaFunc.ECGPU_Mutation[self.ECGPU_BlocksPerGrid2D, self.ECGPU_ThreadsPerBlock2D](self.ECGPU_Chromosomes,
                                                                                 self.ECGPU_MiddleChromosomes,
                                                                                 self.ECGPU_MaxConstraint,
                                                                                 self.ECGPU_MinConstraint,
                                                                                 self.ECArgsDictValueController[
                                                                                     "mutationProbability"],
                                                                                 self.ECArgsDictValueController[
                                                                                     "floatMutationOperateArg"],
                                                                                 self.ECGPU_RngStates2D)
        cuda.synchronize()

    def crossover(self):
        cudaFunc.ECGPU_Crossover[self.ECGPU_BlocksPerGrid1D, self.ECGPU_ThreadsPerBlock1D](self.ECGPU_MidOffspring,
                                                                                  self.ECGPU_MiddleChromosomes,
                                                                                  self.ECArgsDictValueController[
                                                                                      "crossoverRate"],
                                                                                  self.ECGPU_RngStates1D)
        cuda.synchronize()


    def fitting(self, isOffspring=True):
        if isOffspring is False:
            chromosomesDevice = self.ECGPU_Chromosomes
            valDim = self.CHROMOSOME_DIM_INDEX
        else:
            chromosomesDevice = self.ECGPU_MidOffspring
            valDim = self.MIDDLE_CHROMOSOME_DIM_INDEX

        self.fittingInnerForGPU(chromosomesDevice, valDim)

        cudaFunc.ECGPU_CmpToBestChromosomeAndStore[1, 1](
            chromosomesDevice,
            self.ECGPU_BestChromosome,
            self.ECGPU_ChromosomesFittingValue,
            self.ECGPU_ChromosomesAimFuncValue,
            self.ECGPU_BestChromosomesFittingValue,
            self.ECGPU_BestChromosomesAimFuncValue,
            valDim,
            self.BEST_IN_NOW_GEN_DIM_INDEX,
            self.BEST_IN_ALL_GEN_DIM_INDEX)
        cuda.synchronize()

    def fittingInnerForGPU(self, chromosomesDevice, valDim):
        cudaFunc.ECGPU_Fitness[self.ECGPU_BlocksPerGrid1D, self.ECGPU_ThreadsPerBlock1D](chromosomesDevice,
                                                                                self.ECGPU_ChromosomesFittingValue,
                                                                                self.ECGPU_ChromosomesAimFuncValue,
                                                                                self.ECGPU_DataForAimFunc,
                                                                                self.ECArgsDictValueController[
                                                                                    "fittingMinDenominator"],
                                                                                valDim)
        cuda.synchronize()



    def select(self):
        if self.EC_Base_selectType == EC_SelectType.ROULETTE:
            self.selectForRoulette()
        elif self.EC_Base_codingType == EC_SelectType.TOUR:
            self.selectForTour()

    def selectForRoulette(self):
        self.selectPrepareForRoulette()
        self.selectProcessForRoulette()

    def selectForTour(self):
        pass

    def selectPrepare(self):
        cudaFunc.ECGPU_SelectionPrepareCandidates[self.ECGPU_BlocksPerGrid2DSel, self.ECGPU_ThreadsPerBlock2DSel](
            self.ECGPU_Candidates,
            self.ECGPU_Chromosomes,
            self.ECGPU_MidOffspring)

        # cuda.synchronize()

        cudaFunc.ECGPU_SelectionPrepareFitness[self.ECGPU_BlocksPerGrid1DSel, self.ECGPU_ThreadsPerBlock1DSel](
            self.ECGPU_CandidatesFittingValue,
            self.ECGPU_ChromosomesAimFuncValue,
            self.ECGPU_ChromosomesFittingValue,
            self.ECGPU_ChromosomesAimFuncValue,
            self.CHROMOSOME_DIM_INDEX,
            self.MIDDLE_CHROMOSOME_DIM_INDEX, )

        cuda.synchronize()

    def selectPrepareForRoulette(self):
        self.selectPrepare()
        cudaFunc.ECGPU_SelectionPrepareFittingProbForRoulette[1, 1](self.ECGPU_CandidatesFittingValue,
                                                self.ECGPU_FittingProb,
                                                self.ECGPU_CandidateSelectCheckList)
        cuda.synchronize()

    def selectProcessForRoulette(self):
        while True:
            cudaFunc.ECGPU_SelectOneChromosomeForRoulette[self.ECGPU_BlocksPerGrid1DSel, self.ECGPU_ThreadsPerBlock1DSel](
                self.ECGPU_Candidates,
                self.ECGPU_FittingProb,
                self.ECGPU_CandidateSelectCheckList,
                self.ECGPU_RngStates1DSel)
            cuda.synchronize()

            cudaFunc.ECGPU_SelectCheckDoneForRoulette[1, 1](self.Np,
                                        self.ECGPU_CandidateSelectCheckList,
                                        self.ECGPU_CandidateSelectIndexList,
                                        self.ECGPU_SelectDoneFlag,
                                        self.ECGPU_RngStates1DSel)
            cuda.synchronize()

            if self.ECGPU_SelectDoneFlag.copy_to_host()[0] == 1:
                break

        cudaFunc.ECGPU_SelectOffspringForRoulette[self.ECGPU_BlocksPerGrid1D, self.ECGPU_ThreadsPerBlock1D](
            self.ECGPU_Chromosomes,
            self.ECGPU_Candidates,
            self.ECGPU_CandidateSelectIndexList,
            self.ECGPU_CandidatesFittingValue,
            self.ECGPU_CandidatesAimFuncValue,
            self.ECGPU_ChromosomesFittingValue,
            self.ECGPU_ChromosomesAimFuncValue,
            self.CHROMOSOME_DIM_INDEX)
        cuda.synchronize()

    def selectPrepareForTour(self):
        self.selectPrepare()
        cudaFunc.selectionSaveBestChromosome[1, 1](
            self.ECGPU_Chromosomes,
            self.ECGPU_BestChromosome,
            self.ECGPU_ChromosomesFittingValue,
            self.ECGPU_ChromosomesAimFuncValue,
            self.ECGPU_BestChromosomesFittingValue,
            self.ECGPU_BestChromosomesAimFuncValue,
            self.BEST_IN_NOW_GEN_DIM_INDEX,
            self.CHROMOSOME_DIM_INDEX
        )
        cuda.synchronize()

    def selectProcessForTour(self):
        cudaFunc.ECGPU_SelectOffspringForRoulette[self.ECGPU_BlocksPerGrid1D, self.ECGPU_ThreadsPerBlock1D](
            self.ECGPU_Chromosomes,
            self.ECGPU_Candidates,
            self.ECGPU_CandidatesFittingValue,
            self.ECGPU_CandidatesAimFuncValue,
            self.ECGPU_ChromosomesFittingValue,
            self.ECGPU_ChromosomesAimFuncValue,
            self.CHROMOSOME_DIM_INDEX,
            self.ECGPU_RngStates1DSel)
        cuda.synchronize()


    def clearBestChromosome(self, whichBestChromosome):
        if whichBestChromosome == self.BEST_IN_NOW_GEN_DIM_INDEX or whichBestChromosome == self.BEST_IN_ALL_GEN_DIM_INDEX:
            cudaFunc.ECGPU_ClearBestChromosome[1, 1](self.ECGPU_BestChromosomesFittingValue,
                                            self.ECGPU_BestChromosomesAimFuncValue,
                                            whichBestChromosome)
            cuda.synchronize()
        else:
            raise ValueError("input bestChromosome index error")

    def optimizeInner(self):
        super().optimizeInner()
        self.bestChromosome = self.ECGPU_BestChromosome.copy_to_host()
        self.bestChromosomesFittingValue = self.ECGPU_BestChromosomesFittingValue.copy_to_host()
        self.bestChromosomesAimFuncValue = self.ECGPU_BestChromosomesAimFuncValue.copy_to_host()

# import numpy as np
# from numba import cuda
#
# # 定义遗传算法相关的参数
# POPULATION_SIZE = 100  # 种群大小
# NUM_GENERATIONS = 1000  # 迭代次数
# CROSSOVER_RATE = 0.8  # 交叉率
# MUTATION_RATE = 0.1  # 变异率
# NUM_GENES = 10  # 基因数目
# ELITE_SIZE = 10  # 精英数量
#
# # 初始化种群
# @cuda.jit
# def init_population(population):
#     # 通过GPU并行化初始化种群
#     i, j = cuda.grid(2)
#     if i < population.shape[0] and j < population.shape[1]:
#         population[i][j] = np.random.randint(0, 2)
#
# # 计算适应度函数
# @cuda.jit
# def fitness_function(population, fitness):
#     # 通过GPU并行化计算适应度函数
#     i = cuda.grid(1)
#     if i < population.shape[0]:
#         fitness[i] = np.sum(population[i])
#
# # 选择操作
# @cuda.jit
# def selection(population, fitness, offspring):
#     # 通过GPU并行化执行选择操作
#     i, j = cuda.grid(2)
#     if i < offspring.shape[0] and j < offspring.shape[1]:
#         parent1_idx = np.random.randint(0, POPULATION_SIZE)
#         parent2_idx = np.random.randint(0, POPULATION_SIZE)
#         if fitness[parent1_idx] > fitness[parent2_idx]:
#             offspring[i][j] = population[parent1_idx][j]
#         else:
#             offspring[i][j] = population[parent2_idx][j]
#
# @cuda.jit
# def selection(population, fitness, offspring):
#     # 通过GPU并行化执行选择操作
#     i, j = cuda.grid(2)
#     if i < offspring.shape[0] and j < offspring.shape[1]:
#         fitness_sum = np.sum(fitness)
#         threshold = np.random.rand() * fitness_sum
#         cumsum = 0
#         for k in range(POPULATION_SIZE):
#             cumsum += fitness[k]
#             if cumsum >= threshold:
#                 offspring[i][j] = population[k][j]
#                 break
# # 交叉操作
# @cuda.jit
# def crossover(offspring):
#     # 通过GPU并行化执行交叉操作
#     i, j = cuda.grid(2)
#     if i < offspring.shape[0] and j < offspring.shape[1]:
#         if np.random.rand() < CROSSOVER_RATE:
#             parent1_idx = np.random.randint(0, ELITE_SIZE)
#             parent2_idx = np.random.randint(0, POPULATION_SIZE)
#             offspring[i][j] = offspring[parent1_idx][j] if np.random.rand() < 0.5 else offspring[parent2_idx][j]
#
# # 变异操作
# @cuda.jit
# def mutation(offspring):
#     # 通过GPU并行化执行变异操作
#     i, j = cuda.grid(2)
#     if i < offspring.shape[0] and j < offspring.shape[1]:
#         if np.random.rand() < MUTATION_RATE:
#             offspring[i][j] = 1 - offspring[i][j]
#
# # 主函数
# def main():
#     # 初始化CUDA设备
#     cuda.select_device(0)
#
#     # 初始化种群和适应度数组
#     population = cuda.device_array((POPULATION_SIZE, NUM_GENES), dtype=np.int32)
#     fitness = cuda.device_array((POPULATION_SIZE,), dtype=np.int32)
#
#     # 初始化种群
#     init_population[(POPULATION_SIZE, NUM_GENES), (32, 32)](population)
#
#     # 迭代遗传算法
#     for i in range(NUM_GENERATIONS):
#         # 计算适应度函数
#         fitness_function[POPULATION_SIZE, 32 ](population, fitness)
#
#         # 排序种群并选择精英
#         elite_idx = np.argsort(cuda.to_host(fitness))[::-1][:ELITE_SIZE]
#         elite = population[elite_idx]
#
#         # 生成后代并执行遗传算法操作
#         offspring = cuda.device_array((POPULATION_SIZE - ELITE_SIZE, NUM_GENES), dtype=np.int32)
#         selection[(POPULATION_SIZE - ELITE_SIZE, NUM_GENES), (32, 32)](population, fitness, offspring)
#         crossover[(POPULATION_SIZE - ELITE_SIZE, NUM_GENES), (32, 32)](offspring)
#         mutation[(POPULATION_SIZE - ELITE_SIZE, NUM_GENES), (32, 32)](offspring)
#
#         # 合并精英和后代生成新一代种群
#         population[:ELITE_SIZE] = elite
#         population[ELITE_SIZE:] = offspring
#
#     # 计算最优解并返回
#     fitness_function[POPULATION_SIZE, 32](population, fitness)
#     best_idx = np.argmax(cuda.to_host(fitness))
#     best_individual = cuda.to_host(population[best_idx])
#     return best_individual
