#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : UAV_Agent.py
@Author  : jay.zhu
@Time    : 2022/10/13 13:05
"""
import numpy as np
from EC.EC_Common import ArgsDictValueController
from MAS.Agents.UAV_Agent.Agent_UAV_Base import Agent_UAV_Base
from MAS.Agents.UAV_Agent.UAV_Common import calcMovingForUAV, calcDistance


class UAV_Agent(Agent_UAV_Base):
    __UAV_AGENT_DEFAULT_ARGS = {
        "sameBestFittingCountThreshold": 10,
        "bestFittingSameThreshold": 1e-4
    }

    def __init__(self, initPositionState, linearVelocityRange, angularVelocity, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, deltaTime=1., height=1.):
        optimizer = optimizerCls(n=optimizerInitArgs["n"],
                                 dimNum=optimizerInitArgs["dimNum"],
                                 maxConstraint=optimizerInitArgs["maxConstraint"],
                                 minConstraint=optimizerInitArgs["minConstraint"],
                                 evalVars=self.evalVars,
                                 otimizeWay=optimizerInitArgs["otimizeWay"],
                                 needEpochTimes=optimizerInitArgs["needEpochTimes"],
                                 ECArgs=optimizerComputationArgs,
                                 otherTerminalHandler=self.optimizerTerminalHandler)
        super().__init__(initPositionState=initPositionState,
                         linearVelocityRange=linearVelocityRange,
                         angularVelocity=angularVelocity,
                         optimizer=optimizer,
                         deltaTime=deltaTime)

        self.agentArgs = ArgsDictValueController(agentArgs, self.__UAV_AGENT_DEFAULT_ARGS)
        self.sameBestFittingCount = 0
        self.lastBestChromosomesFittingValue = -1
        self.height = height

    def sendMeg(self):
        return self.positionState

    def recvMeg(self, **kwargs):
        self.agentCrowd = np.array(kwargs["agentCrowd"])
        self.selfIndex = kwargs["selfIndex"]
        self.targetPosition = np.array(kwargs["targetPosition"])

    def optimizerTerminalHandler(self, bestChromosomesFittingValue):
        if self.sameBestFittingCount < self.agentArgs["sameBestFittingCountThreshold"]:
            if abs(bestChromosomesFittingValue - self.lastBestChromosomesFittingValue) < self.agentArgs[
                "bestFittingSameThreshold"]:
                self.sameBestFittingCount += 1
            else:
                self.sameBestFittingCount = 0
            self.lastBestChromosomesFittingValue = bestChromosomesFittingValue
            return True
        else:
            return False

    def evalVars(self, chromosome):
        pass

    def evalVars_JTask(self, chromosome):
        newDistanceState = calcMovingForUAV(self.positionState, chromosome, self.deltaTime)
        JTaskVal = np.square(newDistanceState[0] - self.targetPosition[0]) + np.square(
            newDistanceState[1] - self.targetPosition[1])
        return JTaskVal

    def evalVars_JCon(self, chromosome):
        return 0.

    def evalVars_Collision(self, chromosome):
        JCollisionVal = 0.
        for index, item in enumerate(self.agentCrowd):
            if index != self.selfIndex:
                distanceFromItem = calcDistance(self.positionState, item)
                minDistanceThreshold = self.agentArgs["minDistanceThreshold"]
                if distanceFromItem < minDistanceThreshold:
                    JCollisionVal += self.agentArgs["smallDistanceBlameFactor"] * (minDistanceThreshold - distanceFromItem) / minDistanceThreshold

        return JCollisionVal

    def evalVars_Communication(self, chromosome):
        JCommunicationVal = 0.
        for index, item in enumerate(self.agentCrowd):
            if index != self.selfIndex:
                distanceFromItem = calcDistance(self.positionState, item)
                maxDistanceThreshold = self.agentArgs["maxDistanceThreshold"]
                if distanceFromItem > maxDistanceThreshold:
                    JCommunicationVal += self.agentArgs["bigDistanceBlameFactor"] * (distanceFromItem - maxDistanceThreshold) / maxDistanceThreshold

        return JCommunicationVal
