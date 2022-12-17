#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : NashMAS.py
@Author  : jay.zhu
@Time    : 2022/10/14 12:42
"""
import numpy as np
from optimization.common.ArgsDictValueController import ArgsDictValueController


class NashMAS:
    __NASH_MAS_DEFAULT_ARGS = {
        "allCountDiffNashBalanceValue": 5e-1,
        "oneDiffNashBalanceValue": 1e-4
    }

    def __init__(self, agents, masArgs, ):
        super().__init__(agents, masArgs, self.terminalHandler)
        self.lastAgentOptimizationRes = []
        self.NashMas_Args = ArgsDictValueController(masArgs, self.__NASH_MAS_DEFAULT_ARGS, onlyUseDefaultKey=True)



    def terminalHandler(self, agents=None, initFlag=False):
        if initFlag is True:
            self.lastAgentOptimizationRes = []
        else:
            if self.lastAgentOptimizationRes == []:
                self.lastAgentOptimizationRes = [np.array(item.optimizationResult) for item in agents]
                return True
            else:
                continueFlag = False
                for index, item in enumerate(agents):
                    if len(item) <= 5:
                        for index2, itemValue in enumerate(item):
                            if abs(self.lastAgentOptimizationRes[index][index2] - itemValue) > self.NashMas_Args[
                                "oneDiffNashBalanceValue"]:
                                continueFlag = True
                                break
                        if continueFlag == True:
                            break
                    else:
                        AllCountDiff = np.sum(np.abs(np.subtract(self.lastAgentOptimizationRes[index], item)))
                        if AllCountDiff < self.NashMas_Args["allCountDiffNashBalanceValue"]:
                            return False

                if continueFlag == False:
                    return False
                else:
                    for index, item in enumerate(agents):
                        self.lastAgentOptimizationRes[index] = np.array(item)
