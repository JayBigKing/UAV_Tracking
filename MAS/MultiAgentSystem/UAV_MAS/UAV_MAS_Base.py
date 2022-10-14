#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : UAV_MAS_Base.py
@Author  : jay.zhu
@Time    : 2022/10/14 13:58
"""
from MAS.MultiAgentSystem.MAS_Base import MAS_Base


class UAV_MAS_Base(MAS_Base):
    def __init__(self, agents, masArgs, terminalHandler=None):
        super().__init__(agents, masArgs, terminalHandler)
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
