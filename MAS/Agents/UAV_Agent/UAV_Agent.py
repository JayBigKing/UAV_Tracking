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


class UAV_Agent(Agent_UAV_Base):
    __UAV_AGENT_DEFAULT_ARGS = {
        "predictVelocityLen": 1,
        "usePredictVelocityLen": 1,
        "sameBestFittingCountThreshold": 10,
        "fittingIsSameThreshold": 1e-4,
        "JTaskFactor": .9,
        "JConFactor": .1,
        "JColFactor": 1.,
        "JComFactor": 1.,
        "minDistanceThreshold": 9.,
        "smallDistanceBlameFactor": 19.,
        "maxDistanceThreshold": 30.,
        "bigDistanceBlameFactor": 15.,
    }

    def __init__(self, initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, deltaTime=1., height=1.):
        self.agentArgs = ArgsDictValueController(agentArgs, self.__UAV_AGENT_DEFAULT_ARGS)
        self.predictVelocityLen = self.agentArgs["predictVelocityLen"]

        optimizerInitArgs["optimizeWay"] = EC_OtimizeWay.MIN
        optimizerInitArgs["maxConstraint"] = []
        optimizerInitArgs["minConstraint"] = []
        optimizerInitArgs["optimizerLearningDimNum"] = optimizerInitArgs["dimNum"] * self.predictVelocityLen

        for i in range(self.predictVelocityLen):
            optimizerInitArgs["maxConstraint"].append(linearVelocityRange[1])
            optimizerInitArgs["maxConstraint"].append(angularVelocityRange[1])
            optimizerInitArgs["minConstraint"].append(linearVelocityRange[0])
            optimizerInitArgs["minConstraint"].append(angularVelocityRange[0])
            optimizerComputationArgs["borders"].append(optimizerComputationArgs["borders"][0])
            optimizerComputationArgs["borders"].append(optimizerComputationArgs["borders"][1])

        optimizer = optimizerCls(n=optimizerInitArgs["n"],
                                 dimNum=optimizerInitArgs["optimizerLearningDimNum"],
                                 maxConstraint=optimizerInitArgs["maxConstraint"],
                                 minConstraint=optimizerInitArgs["minConstraint"],
                                 evalVars=self.evalVars,
                                 otimizeWay=optimizerInitArgs["optimizeWay"],
                                 needEpochTimes=optimizerInitArgs["needEpochTimes"],
                                 ECArgs=optimizerComputationArgs,
                                 otherTerminalHandler=self.optimizerTerminalHandler)
        super().__init__(initPositionState=initPositionState,
                         linearVelocityRange=linearVelocityRange,
                         angularVelocityRange=angularVelocityRange,
                         optimizer=optimizer,
                         deltaTime=deltaTime)

        self.predictVelocityList = np.zeros(optimizerInitArgs["optimizerLearningDimNum"])
        self.sameBestFittingCount = 0
        self.lastBestChromosomesFittingValue = -1
        self.height = height
        self.remainMoving = 0

    def sendMeg(self):
        return self.positionState, self.velocity, self.predictVelocityList

    def recvMeg(self, **kwargs):
        self.agentCrowd = kwargs["agentCrowd"]
        self.selfIndex = kwargs["selfIndex"]
        self.targetPositionList = self.predictTargetMoving(kwargs["targetPosition"])

        # self.agentCrowdPositionState = self.agentCrowd["positionState"]
        # self.agentCrowdVelocity = self.agentCrowd["positionState"]["velocity"]
        # self.agentCrowdOptimizationResult = self.agentCrowd["positionState"]["optimizationResult"]
        self.optimizer.ECDynOptHyperMutation_ECArgsDictValueController["performanceThreshold"] = 1 / (
                calcDistance(self.positionState, self.targetPositionList[0]) *
                self.optimizer.ECArgsDictValueController["fittingMinDenominator"])

    def predictTargetMoving(self, targetPosition):
        if isinstance(targetPosition, list) is False:
            return [np.array(targetPosition) for i in range(self.predictVelocityLen)]
        else:
            return targetPosition

    def optimization(self):
        if self.remainMoving == 0:
            super().optimization()
            self.predictVelocityList = self.optimizationResult[0]
            # self.remainMoving = self.agentArgs["usePredictVelocityLen"]
        for item in self.statFuncReg:
            item(optimizationResult=self.optimizationResult)

    def updateInner(self):
        if self.remainMoving == 0:
            self.remainMoving = self.agentArgs["usePredictVelocityLen"]
        super().updateInner()
        self.remainMoving -= 1

    def moving(self):
        startIndex, endIndex = self.getVelocityFromPredictVelocityList(
            self.agentArgs["usePredictVelocityLen"] - self.remainMoving)
        self.positionState = calcMovingForUAV(self.positionState, self.predictVelocityList[startIndex: endIndex],
                                              self.deltaTime)

    def optimizerTerminalHandler(self, bestChromosomesFittingValue=0, initFlag=False):
        if initFlag:
            self.sameBestFittingCount = 0
        else:
            if self.sameBestFittingCount < self.agentArgs["sameBestFittingCountThreshold"]:
                if abs(bestChromosomesFittingValue - self.lastBestChromosomesFittingValue) < self.agentArgs[
                    "fittingIsSameThreshold"]:
                    self.sameBestFittingCount += 1
                else:
                    self.sameBestFittingCount = 0
                self.lastBestChromosomesFittingValue = bestChromosomesFittingValue
                return True
            else:
                return False

    def getVelocityFromPredictVelocityList(self, velocityIndex):
        velocityLen = self.velocity.size
        return int((velocityIndex) * velocityLen), int((velocityIndex + 1) * velocityLen)

    def evalVars(self, chromosome):
        # return self.agentArgs["JTaskFactor"] * self.evalVars_JTask(chromosome, self.predictVelocityLen) + \
        #        self.agentArgs["JConFactor"] * self.evalVars_JConsume(chromosome,self.predictVelocityLen) + \
        #        self.agentArgs["JColFactor"] * self.evalVars_JCollision(chromosome, self.predictVelocityLen) + \
        #        self.agentArgs["JComFactor"] * self.evalVars_JCommunication(chromosome, self.predictVelocityLen)
        return self.agentArgs["JTaskFactor"] * self.evalVars_JTask(chromosome,
                                                                   self.predictVelocityLen) + \
               self.agentArgs["JConFactor"] * self.evalVars_JConsume(chromosome, self.predictVelocityLen)

    def evalVars_JTask(self, chromosome, *args):
        predictVelocityLen = args[0]
        JTaskList = np.zeros(predictVelocityLen)
        oldPositionList = [self.positionState]
        originDisFromTarget = 0.

        for i in range(predictVelocityLen):
            startIndex, endIndex = self.getVelocityFromPredictVelocityList(i)
            newPositionState = calcMovingForUAV(oldPositionList[i], chromosome[startIndex: endIndex], self.deltaTime)
            targetPosition = self.targetPositionList[i]
            JTaskList[i] = np.square(newPositionState[0] - targetPosition[0]) + np.square(
                newPositionState[1] - targetPosition[1])
            if i == 0:
                originDisFromTarget = np.square(self.positionState[0] - targetPosition[0]) + np.square(
                    self.positionState[1] - targetPosition[1])
            oldPositionList.append(newPositionState)
        JTaskMax = max(np.max(JTaskList), originDisFromTarget * 2.)
        JTaskVal = ((np.average(JTaskList)) / (JTaskMax))
        return JTaskVal

    def evalVars_JConsume(self, chromosome, *args):
        predictVelocityLen = args[0]
        JConsumeList = np.zeros((2, predictVelocityLen - 1))
        for i in range(predictVelocityLen - 1):
            startIndex, endIndex = self.getVelocityFromPredictVelocityList(i)
            JConsumeList[0, i] = abs(chromosome[startIndex] - chromosome[endIndex])
            JConsumeList[1, i] = abs(chromosome[startIndex + 1] - chromosome[endIndex + 1])

        # JConsumeMax = np.max(JConsumeList, axis=1)
        # JConsumeMin = np.min(JConsumeList, axis=1)
        JConsumeAvg = np.average(JConsumeList, axis=1)
        JConsumeVal = 0.5 * ((JConsumeAvg[0] - self.linearVelocityRange[0]) / (self.linearVelocityRange[1] - self.linearVelocityRange[0]) + (
                    abs(JConsumeAvg[1])) / (self.angularVelocityRange[1]))
        return JConsumeVal

    def evalVars_JCollision(self, chromosome, *args):
        JCollisionVal = 0.
        predictVelocityLen = args[0]

        for index, item in enumerate(self.agentCrowd["positionState"]):
            if index != self.selfIndex:
                oldPositionList = self.positionState
                oldPositionListForItem = item
                chromosomeForItem = self.agentCrowd["predictVelocityList"][index]
                JCollisionList = np.zeros(predictVelocityLen)

                for i in range(predictVelocityLen):
                    startIndex, endIndex = self.getVelocityFromPredictVelocityList(i)
                    newPositionState = calcMovingForUAV(oldPositionList, chromosome[startIndex: endIndex],
                                                        self.deltaTime)
                    newPositionStateForItem = calcMovingForUAV(oldPositionListForItem,
                                                               chromosomeForItem[startIndex: endIndex],
                                                               self.deltaTime)
                    JCollisionList[i] = calcDistance(newPositionState[0: 2], newPositionStateForItem[0: 2])
                    oldPositionList = newPositionState
                    oldPositionListForItem = newPositionStateForItem

                distanceFromItem = np.average(JCollisionList)
                minDistanceThreshold = self.agentArgs["minDistanceThreshold"]
                if distanceFromItem < minDistanceThreshold:
                    JCollisionVal += self.agentArgs["smallDistanceBlameFactor"] * (
                            minDistanceThreshold - distanceFromItem) / minDistanceThreshold

        return JCollisionVal

    def evalVars_JCommunication(self, chromosome, *args):
        # return 0.
        JCommunicationVal = 0.
        newPositionState = calcMovingForUAV(self.positionState, chromosome, self.deltaTime)
        for index, item in enumerate(self.agentCrowd["positionState"]):
            if index != self.selfIndex:
                distanceFromItem = calcDistance(newPositionState[0:2], item[0:2])
                maxDistanceThreshold = self.agentArgs["maxDistanceThreshold"]
                if distanceFromItem > maxDistanceThreshold:
                    JCommunicationVal += self.agentArgs["bigDistanceBlameFactor"] + self.agentArgs[
                        "bigDistanceBlameFactor"] * Jay_sigmoid(
                        distanceFromItem - maxDistanceThreshold) + 2.

        return JCommunicationVal
