#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : testGWO.py
@Author  : jay.zhu
@Time    : 2023/3/10 19:44
"""
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from optimization.common.optimizationCommonEnum import OptimizationWay
from optimization.GWO.GWO_Base import GWO_Base
from optimization.GWO.GWO_IGWO_PENG import GWO_IGWO_PENG

def evalFunc1(chromosome):
    return (chromosome[0] - 1) ** 2

@clockTester
def test1CPU():
    gwo = GWO_Base(100, 1, [100.], [-100.], evalFunc1, OptimizationWay.MIN, 100, {"borders": [1]})
    chromosome, bestVal, _ = gwo.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))

@clockTester
def test1IGWO():
    gwo = GWO_IGWO_PENG(100, 1, [100.], [-100.], evalFunc1, OptimizationWay.MIN, 100, {"borders": [1]})
    chromosome, bestVal, _ = gwo.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))

def main():
    test1IGWO()

main()