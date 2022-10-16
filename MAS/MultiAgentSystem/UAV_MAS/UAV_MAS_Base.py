#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : UAV_MAS_Base.py
@Author  : jay.zhu
@Time    : 2022/10/14 13:58
"""
import numpy as np
from MAS.MultiAgentSystem.MAS_WithStat_Base import MAS_WithStat_Base
from MAS.Agents.UAV_Agent.UAV_Common import calcDistance

class UAV_MAS_Base(MAS_WithStat_Base):
    def __init__(self, agents, masArgs, terminalHandler=None):
        super().__init__(agents, masArgs, terminalHandler, [self.UAV_MAS_Stat_recordDisOfUAVs])
        self.lastAgentOptimizationRes = []

    def optimizationPreProcess(self):
        agentsPositionState = []
        agentsVelocity = []
        agentsOptimizationResult = []

        for item in self.agents:
            agentsPositionState.append(item.positionState)
            agentsVelocity.append(item.velocity)
            agentsOptimizationResult.append(item.optimizationResult)

        self.agentCrowd = {
            "positionState": agentsPositionState,
            "velocity": agentsVelocity,
            "optimizationResult": agentsOptimizationResult
        }

        for index, item in enumerate(self.agents):
            item.recvMeg(agentCrowd=self.agentCrowd, selfIndex=index, targetPosition=self.targetPosition)

    def recvFromEnv(self, **kwargs):
        self.targetPosition = kwargs["targetPosition"]

    '''
    following is stat function
    '''
    def UAV_MAS_Stat_recordDisOfUAVs(self, **kwargs):
        agentLen = len(self.agents)
        if hasattr(self, 'UAVDisMatrix') is False:
            self.UAVDisMatrix = np.zeros((agentLen, agentLen))
        if hasattr(self, 'UAVDisStatMatrix') is False:
            self.UAVDisStatMatrix = np.zeros((agentLen, agentLen, 2))
            for i in range(agentLen):
                for j in range(agentLen):
                    if i != j:
                        self.UAVDisStatMatrix[i, j, 1] = 1e10


        for index, item in enumerate(self.agents):
            for j in range(agentLen):
                if index != j:
                    if j < index:
                        self.UAVDisMatrix[index, j] = self.UAVDisMatrix[j, index]
                    else:
                        self.UAVDisMatrix[index, j] = calcDistance(item.positionState[0: 2], self.agents[j].positionState[0: 2])

                    self.UAVDisStatMatrix[index, j, 0] += self.UAVDisMatrix[index, j]
                    self.UAVDisStatMatrix[index, j, 1] = min(self.UAVDisMatrix[index, j], self.UAVDisStatMatrix[index, j, 1])

        print(self.UAVDisMatrix)



