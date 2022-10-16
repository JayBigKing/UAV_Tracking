#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : UAV_NashMAS.py
@Author  : jay.zhu
@Time    : 2022/10/14 14:09
"""
import numpy as np
from MAS.MultiAgentSystem.UAV_MAS.UAV_MAS_Base import UAV_MAS_Base,ArgsDictValueController


class UAV_NashMAS(UAV_MAS_Base):
    __NASH_MAS_DEFAULT_ARGS = {
        "allCountDiffNashBalanceValue": 5e-1,
        "oneDiffNashBalanceValue": 1e-4
    }

    def __init__(self, agents, masArgs, ):
        super().__init__(agents, masArgs, self.terminalHandler)
        # super().__init__(agents, masArgs, None)
        self.lastAgentOptimizationRes = []
        self.NashMas_Args = ArgsDictValueController(masArgs, self.__NASH_MAS_DEFAULT_ARGS, onlyUseDefaultKey=True)

    def communication(self):
        predictVelocityList = [item.predictVelocityList for item in self.agents]

        self.agentCrowd["predictVelocityList"] = predictVelocityList

        for index, item in enumerate(self.agents):
            item.recvMeg(agentCrowd=self.agentCrowd, selfIndex=index, targetPosition=self.targetPosition)

    def terminalHandler(self, agents=None, initFlag=False):
        #continue返回True
        #否则返回False
        if initFlag is True:
            self.lastAgentOptimizationRes = []
        else:
            if self.lastAgentOptimizationRes == []:
                # if agents[0].optimizationResult is not None:
                self.lastAgentOptimizationRes = [item.predictVelocityList for item in agents]
                return True
            else:
                continueFlag = False
                for index, item in enumerate(agents):
                    if len(item.predictVelocityList) <= 0:
                        for index2, itemValue in enumerate(item.predictVelocityList):
                            if abs(self.lastAgentOptimizationRes[index][index2] - itemValue) > self.NashMas_Args[
                                "oneDiffNashBalanceValue"]:
                                continueFlag = True
                                break
                        if continueFlag == True:
                            break
                    else:
                        AllCountDiff = np.sum(np.abs(self.lastAgentOptimizationRes[index] - item.predictVelocityList))
                        if AllCountDiff > self.NashMas_Args["allCountDiffNashBalanceValue"]:
                            continueFlag = True
                            break

                if continueFlag == False:
                    return False
                else:
                    for index, item in enumerate(agents):
                        self.lastAgentOptimizationRes[index] = np.array(item.predictVelocityList)
                    return True
