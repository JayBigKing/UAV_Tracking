#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : testCudaEC.py
@Author  : jay.zhu
@Time    : 2023/2/26 12:32
"""
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from optimization.common.optimizationCommonEnum import OptimizationWay
from numba import cuda
from optimization.EC.EC_Base import EC_SelectType
from optimization.EC.EC_Base import EC_Base
from optimization.EC.GPU.ECGPU_Base import ECGPU_Base


@cuda.jit(device=True)
def evalFunc1GPU(chromosomes, idx, dataForAimFunc):
    return (chromosomes[0, idx] - 1) ** 2


def evalFunc1(chromosome):
    return (chromosome[0] - 1) ** 2


@clockTester
def test1GPU():
    eb = ECGPU_Base(n=100, dimNum=1, maxConstraint=[100.], minConstraint=[-100.],
                    evalVars=evalFunc1GPU, otimizeWay=OptimizationWay.MIN, needEpochTimes=100, ECArgs={"borders": [1],
                                                                                                       "dataForAimFuncSize": 1,
                                                                                                       "EC_ChoosingType": EC_SelectType.TOUR})
    chromosome, bestVal, _ = eb.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))


@clockTester
def test1CPU():
    eb = EC_Base(100, 1, [100.], [-100.], evalFunc1, OptimizationWay.MIN, 100, {"borders": [1]})
    chromosome, bestVal, _ = eb.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))


@cuda.jit(device=True)
def evalFunc2GPU(chromosomes, idx, dataForAimFunc):
    count = 0.
    for j in range(chromosomes.shape[0]):
        count += chromosomes[j, idx]
    for j in range(2, 4):
        count -= chromosomes[j, idx] * chromosomes[j, idx]
    return count ** 2


def evalFunc2(chromosome):
    count = 0.
    for j in range(chromosome.shape[0]):
        count += chromosome[j]
    for j in range(2, 4):
        count -= chromosome[j] * chromosome[j]

    return count ** 2


@clockTester
def test2GPU():
    n = 3000
    dimNum = 7
    eb = ECGPU_Base(n=n, dimNum=dimNum, maxConstraint=[100. for i in range(dimNum)],
                    minConstraint=[-100. for i in range(dimNum)],
                    evalVars=evalFunc2GPU, otimizeWay=OptimizationWay.MIN, needEpochTimes=100,
                    ECArgs={"borders": [1 for i in range(dimNum)],
                            "dataForAimFuncSize": 1,
                            "EC_ChoosingType": EC_SelectType.TOUR})
    chromosome, bestVal, _ = eb.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))


@clockTester
def test2CPU():
    n = 3000
    dimNum = 7
    eb = EC_Base(n, dimNum, [100. for i in range(dimNum)], [-100. for i in range(dimNum)], evalFunc2,
                 OptimizationWay.MIN, 100, {"borders": [1 for i in range(dimNum)],
                                            "EC_ChoosingType": EC_SelectType.TOUR})
    chromosome, bestVal, _ = eb.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))


def main():
    test2GPU()
    test2CPU()


if __name__ == "__main__":
    main()
