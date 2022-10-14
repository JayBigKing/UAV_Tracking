#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : UAV_Agent.py
@Author  : jay.zhu
@Time    : 2022/10/13 13:05
"""
import numpy as np
from algorithmTool.mathFunction.mathFunction import Jay_sigmoid
from EC.EC_Common import ArgsDictValueController
from EC.EC_Base import EC_OtimizeWay
from MAS.Agents.UAV_Agent.Agent_UAV_Base import Agent_UAV_Base
from MAS.Agents.UAV_Agent.UAV_Common import calcMovingForUAV, calcDistance
from copy import deepcopy


class UAV_Agent(Agent_UAV_Base):
    __UAV_AGENT_DEFAULT_ARGS = {
        "sameBestFittingCountThreshold": 10,
        "bestFittingSameThreshold": 1e-4,
        "JTaskFactor": 1.,
        "JConsumeFactor": 1.,
        "JCollisionFactor": 1.,
        "JCommunicationFactor": 1.,

    }

    def __init__(self, initPositionState, linearVelocityRange, angularVelocity, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, deltaTime=1., height=1.):
        optimizerInitArgs["maxConstraint"] = [linearVelocityRange[1], angularVelocity[1]]
        optimizerInitArgs["minConstraint"] = [linearVelocityRange[0], angularVelocity[0]]
        optimizerInitArgs["otimizeWay"] = EC_OtimizeWay.MIN
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
        return self.positionState, self.velocity, self.optimizationResult

    def recvMeg(self, **kwargs):
        self.agentCrowd = kwargs["agentCrowd"]
        self.selfIndex = kwargs["selfIndex"]
        self.targetPosition = np.array(kwargs["targetPosition"])

        # self.agentCrowdPositionState = self.agentCrowd["positionState"]
        # self.agentCrowdVelocity = self.agentCrowd["positionState"]["velocity"]
        # self.agentCrowdOptimizationResult = self.agentCrowd["positionState"]["optimizationResult"]
        self.optimizer.ECDynOptHyperMutation_ECArgsDictValueController["performanceThreshold"] = 1 / (
                    calcDistance(self.positionState, self.targetPosition) *
                    self.optimizer.ECArgsDictValueController["fittingMinDenominator"])

    def optimization(self):
        super().optimization()
        self.predictVelocity = self.optimizationResult[0]


    def optimizerTerminalHandler(self, bestChromosomesFittingValue = 0, initFlag = False):
        if initFlag:
            self.sameBestFittingCount = 0
        else:
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
        # return self.agentArgs["JTaskFactor"] * self.evalVars_JTask(chromosome) + \
        #        self.agentArgs["JConsumeFactor"] * self.evalVars_JConsume(chromosome) + \
        #        self.agentArgs["JCollisionFactor"] * self.evalVars_JCollision(chromosome) + \
        #        self.agentArgs["JCommunicationFactor"] * self.evalVars_JCommunication(chromosome)
        return self.agentArgs["JTaskFactor"] * self.evalVars_JTask(chromosome)

    def evalVars_JTask(self, chromosome):
        newPositionState = calcMovingForUAV(self.positionState, chromosome, self.deltaTime)
        JTaskVal = np.square(newPositionState[0] - self.targetPosition[0]) + np.square(
            newPositionState[1] - self.targetPosition[1])
        return JTaskVal

    def evalVars_JConsume(self, chromosome):
        return 0.

    def evalVars_JCollision(self, chromosome):
        JCollisionVal = 0.
        newPositionState = calcMovingForUAV(self.positionState, chromosome, self.deltaTime)
        for index, item in enumerate(self.agentCrowd["positionState"]):
            if index != self.selfIndex:

                distanceFromItem = calcDistance(newPositionState, item)
                minDistanceThreshold = self.agentArgs["minDistanceThreshold"]
                if distanceFromItem < minDistanceThreshold:
                    JCollisionVal += self.agentArgs["smallDistanceBlameFactor"] * (
                                minDistanceThreshold - distanceFromItem) / minDistanceThreshold

        return JCollisionVal

    def evalVars_JCommunication(self, chromosome):
        JCommunicationVal = 0.
        newPositionState = calcMovingForUAV(self.positionState, chromosome, self.deltaTime)
        for index, item in enumerate(self.agentCrowd["positionState"]):
            if index != self.selfIndex:
                distanceFromItem = calcDistance(newPositionState, item)
                maxDistanceThreshold = self.agentArgs["maxDistanceThreshold"]
                if distanceFromItem > maxDistanceThreshold:
                    # JCommunicationVal += self.agentArgs["bigDistanceBlameFactor"] * (distanceFromItem - maxDistanceThreshold) / maxDistanceThreshold
                    JCommunicationVal += self.agentArgs["bigDistanceBlameFactor"] * Jay_sigmoid(
                        distanceFromItem - maxDistanceThreshold) + 2.

        return JCommunicationVal
