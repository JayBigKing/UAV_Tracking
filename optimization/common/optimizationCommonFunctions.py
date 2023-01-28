#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : optimizationCommonFunctions.py
@Author  : jay.zhu
@Time    : 2022/12/15 16:43
"""
from optimization.common.optimizationCommonEnum import OptimizationWay

def shouldContinue(nowEpochTime, bestFittingValue, bestAimFuncValue, needEpochTimes, otherTerminalHandler=None):
    nowEpochTime += 1
    if otherTerminalHandler is not None:
        if otherTerminalHandler(bestFittingValue=bestFittingValue, bestAimFuncValue=bestAimFuncValue) is False:
            return False, nowEpochTime

    if nowEpochTime <= needEpochTimes:
        return True, nowEpochTime
    else:
        return False, nowEpochTime


def limitValue(value, dimIndex, maxConstraint, minConstraint):
    return min(maxConstraint[dimIndex], max(value, minConstraint[dimIndex]))

def callAimFunc(solution, evalVars):
    return evalVars(solution)

def fittingOne(solution, evalVars, optimizeWay, fittingMinDenominator):
    aimFuncVal = callAimFunc(solution, evalVars)
    fittingValue = aimFuncVal
    if optimizeWay == OptimizationWay.MIN:
        fittingMinDenominator = fittingMinDenominator
        fittingValue = 1 / (fittingValue + fittingMinDenominator)
    return fittingValue, aimFuncVal

def cmpFitting(val1, val2):
    return (val1 - val2)
