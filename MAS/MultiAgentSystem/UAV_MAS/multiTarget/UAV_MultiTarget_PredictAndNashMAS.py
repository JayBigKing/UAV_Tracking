#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_PredictAndNashMAS.py
@Author  : jay.zhu
@Time    : 2022/11/5 20:05
"""
import numpy as np
from copy import deepcopy
from optimization.common.ArgsDictValueController import ArgsDictValueController
from multiprocessing import Process, Manager, Semaphore
from MAS.MultiAgentSystem.MAS_MultiThread_Base import MAS_MultiThread_Base
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_PredictMAS import UAV_MultiTarget_PredictMAS


class UAV_MultiTarget_PredictAndNashMAS(UAV_MultiTarget_PredictMAS):
    __UAV_MULTITARGET_PREDICTANDNASHMAS_DEFAULT_ARGS = {
        "allCountDiffNashBalanceValue": 5e-1,
        "oneDiffNashBalanceValue": 1e-4,
        "usingMultiThread": False
    }

    def __init__(self, agents, masArgs, targetNum, terminalHandler=None, predictorCls=None, statRegisters=None,
                 deltaTime=1.):
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

    def PredictAndNashMAS_CallAgentProcess(self, agent, index, argDict, managerDict, loadAgentSem, killProSem):
        if argDict.get("init") is not None and argDict["init"] is True:
            agent.optimization(init=argDict["init"])
        else:
            agent.optimization()

        managerDict[index] = deepcopy(agent)
        loadAgentSem.release()
        killProSem.acquire()


    def PredictAndNashMAS_OptimizationInnerForAgents(self, **kwargs):
        if self.NashMas_Args["usingMultiThread"] is False:
            for agent in self.agents:
                agent.optimization(**kwargs)
        else:
            with Manager() as manager:
                dictList = [manager.dict() for i in self.agents]
                loadAgentSem = Semaphore(len(self.agents))
                killProSemList = [Semaphore(1) for i in self.agents]

                for i in range(len(self.agents)):
                    loadAgentSem.acquire()

                processList = [
                    Process(target=self.PredictAndNashMAS_CallAgentProcess,
                            args=(agent, index, {"init": kwargs["init"] if kwargs.get("init") is not None else False},
                                  dictList[index], loadAgentSem, killProSemList[index])) for
                    index, agent in enumerate(self.agents)]

                for p in processList:
                    p.start()

                for i in range(len(self.agents)):
                    loadAgentSem.acquire()

                for index, newAgent in enumerate(dictList):
                    self.agents[index] = newAgent[index]

                for killProSem in killProSemList:
                    killProSem.release()

            manager.join()

    def optimizationInner(self):
        self.communication()
        if self.firstNashOptimizationFlag is True:
            self.PredictAndNashMAS_OptimizationInnerForAgents(init=True)
            self.firstNashOptimizationFlag = False
        else:
            self.PredictAndNashMAS_OptimizationInnerForAgents()

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
