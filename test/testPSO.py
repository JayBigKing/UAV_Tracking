#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : testPSO.py
@Author  : jay.zhu
@Time    : 2022/12/15 17:58
"""
import random
import numpy as np
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from optimization.PSO.PSO_Base import PSO_Base, OptimizationWay


def evalFunc1(chromosome):
    return (chromosome[0] - 1) ** 2

evalFunc2Count = 0
evalFunc2Val = 2.
def evalFunc2(chromosome):
    global evalFunc2Count
    global evalFunc2Val
    if evalFunc2Count == 0:
        evalFunc2Val = float(random.randint(1, 10))
        print('evalFunc2Val is %f'%evalFunc2Val)
    elif evalFunc2Count == 10 * 30:
        evalFunc2Val = float(random.randint(-60, -10))
        print('evalFunc2Val is %f'%evalFunc2Val)

    evalFunc2Count += 1

    return (2 * chromosome[0] - evalFunc2Val) ** 2

def evalFunc3(solution):
    return np.sin(solution[0]) + 1

@clockTester
def test1():
    pso = PSO_Base(n=100, dimNum=1, positionMaxConstraint=[100.], positionMinConstraint=[-100.],
                       evalVars=evalFunc1, optimizeWay=OptimizationWay.MAX ,needEpochTimes=100, PSOArgs={"borders":[1]})
    chromosome, bestVal = pso.optimization()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))

@clockTester
def test2():
    pso = PSO_Base(n=100, dimNum=1, positionMaxConstraint=[0.], positionMinConstraint=[2 * np.pi],
                       evalVars=evalFunc3, optimizeWay=OptimizationWay.MAX ,needEpochTimes=100, PSOArgs={"borders":[1]})
    chromosome, bestVal = pso.optimization()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))

def main():
    test2()

if __name__ == "__main__":
    main()