#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_PredictAndNashMAS.py
@Author  : jay.zhu
@Time    : 2022/11/5 20:05
"""
import numpy as np
from optimization.common.ArgsDictValueController import ArgsDictValueController
from MAS.MultiAgentSystem.MAS_MultiThread_Base import MAS_MultiThread_Base
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_PredictMAS import UAV_MultiTarget_PredictMAS

class UAV_MultiTarget_PredictAndNashMAS(UAV_MultiTarget_PredictMAS):
    __UAV_MULTITARGET_PREDICTANDNASHMAS_DEFAULT_ARGS = {
        "allCountDiffNashBalanceValue": 5e-1,
        "oneDiffNashBalanceValue": 1e-4
    }
    def __init__(self, agents, masArgs, targetNum, terminalHandler=None, predictorCls=None, statRegisters=None, deltaTime=1.):
        super().__init__(agents=agents,
                         masArgs=masArgs,
                         targetNum=targetNum,
                         terminalHandler=self.terminalHandler,
                         predictorCls=predictorCls,
                         statRegisters=statRegisters,
                         deltaTime=deltaTime)
        self.lastAgentOptimizationRes = []
        self.NashMas_Args = ArgsDictValueController(masArgs,
                                                    self.__UAV_MULTITARGET_PREDICTANDNASHMAS_DEFAULT_ARGS,
                                                    onlyUseDefaultKey=True)
        self.firstCommunicationFlag = True

    def initShouldContinueOptimizationVar(self, terminalHandler):
        super().initShouldContinueOptimizationVar(terminalHandler)
        self.firstCommunicationFlag = True

    def optimization(self):
        self.firstNashOptimizationFlag = True
        super().optimization()


    def optimizationInner(self):
        self.communication()
        if self.firstNashOptimizationFlag is True:
            for agent in self.agents:
                agent.optimization(init=True)
            self.firstNashOptimizationFlag = False
        else:
            for agent in self.agents:
                agent.optimization()

    def communication(self):
        if self.firstCommunicationFlag is True:
            self.firstCommunicationFlag = False
        predictVelocityList = [item.predictVelocityList for item in self.agents]
        self.agentCrowd["predictVelocityList"] = predictVelocityList
        numOfTrackingUAVForTargetList = np.zeros(self.targetNum)
        for item in predictVelocityList:
            numOfTrackingUAVForTargetList[int(item[0])] += 1.


        for index, item in enumerate(self.agents):
            item.recvMeg(agentCrowd=self.agentCrowd, selfIndex=index,
                         targetPositionList=self.targetPositionList,
                         numOfTrackingUAVForTargetList=numOfTrackingUAVForTargetList)


    def terminalHandler(self, agents=None, initFlag=False):
        if initFlag is True:
            self.lastAgentOptimizationRes = []
        else:
            if self.lastAgentOptimizationRes == []:
                if agents[0].optimizationResult is not None:
                    self.lastAgentOptimizationRes = [item.predictVelocityList for item in agents]
                return True
            else:
                continueFlag = False
                for index, item in enumerate(agents):
                    if len(item.optimizationResult) <= 0:
                        for index2, itemValue in enumerate(item.predictVelocityList):
                            if abs(self.lastAgentOptimizationRes[index][index2] - itemValue) > self.NashMas_Args[
                                "oneDiffNashBalanceValue"]:
                                continueFlag = True
                                break
                        if continueFlag == True:
                            break
                    else:
                        continueFlag = True
                        AllCountDiff = np.sum(np.abs(self.lastAgentOptimizationRes[index] - item.predictVelocityList))
                        if AllCountDiff < self.NashMas_Args["allCountDiffNashBalanceValue"]:
                            return False

                if continueFlag == False:
                    return False
                else:
                    for index, item in enumerate(agents):
                        self.lastAgentOptimizationRes[index] = item.predictVelocityList

