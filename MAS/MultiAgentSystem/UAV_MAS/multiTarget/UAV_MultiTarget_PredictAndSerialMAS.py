#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_PredictAndSerialMAS.py
@Author  : jay.zhu
@Time    : 2022/11/6 13:42
"""
import numpy as np
from MAS.MultiAgentSystem.MAS_SerialBase import MAS_SerialBase
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_PredictMAS import UAV_MultiTarget_PredictMAS


class UAV_MultiTarget_PredictAndSerialMAS(UAV_MultiTarget_PredictMAS, MAS_SerialBase):
    def __init__(self, agents, masArgs, targetNum, terminalHandler=None, predictorCls=None, deltaTime=1.):
        UAV_MultiTarget_PredictMAS.__init__(self,
                                            agents=agents,
                                            masArgs=masArgs,
                                            targetNum=targetNum,
                                            terminalHandler=terminalHandler,
                                            predictorCls=predictorCls,
                                            deltaTime=deltaTime)
        self.optimizationAgentIndexCount = 0
        self.middleNumOfTrackingUAVForTargetList = []

    def initShouldContinueOptimizationVar(self, terminalHandler):
        super().initShouldContinueOptimizationVar(terminalHandler)
        self.optimizationAgentIndexCount = 0

    def optimizationInner(self):
        self.middleNumOfTrackingUAVForTargetList = np.array(self.numOfTrackingUAVForTargetList)
        super().optimizationInner()

    def communication(self):
        nowOptimizationAgent = self.agents[self.optimizationAgentIndexCount]
        originTargetIndex = nowOptimizationAgent.trackingTargetIndex
        newTargetIndex = int(nowOptimizationAgent.predictVelocityList[0])

        self.agentCrowd["predictVelocityList"][
            self.optimizationAgentIndexCount] = nowOptimizationAgent.predictVelocityList

        if originTargetIndex != newTargetIndex:
            self.middleNumOfTrackingUAVForTargetList[newTargetIndex] += 1.
            if self.middleNumOfTrackingUAVForTargetList[originTargetIndex] >= 1.:
                self.middleNumOfTrackingUAVForTargetList[originTargetIndex] -= 1

        for index in range(self.optimizationAgentIndexCount + 1, len(self.agents)):
            self.agents[index].recvMeg(agentCrowd=self.agentCrowd, selfIndex=index,
                                       targetPositionList=self.targetPositionList,
                                       numOfTrackingUAVForTargetList=self.middleNumOfTrackingUAVForTargetList)

        self.optimizationAgentIndexCount += 1
