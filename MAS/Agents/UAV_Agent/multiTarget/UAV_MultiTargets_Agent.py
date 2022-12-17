#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_Agent.py
@Author  : jay.zhu
@Time    : 2022/11/4 20:03
"""
import numpy as np

from MAS.Agents.UAV_Agent.UAV_Common import calcMovingForUAV, calcDistance
from MAS.Agents.UAV_Agent.UAV_Agent import UAV_Agent


class UAV_MultiTarget_Agent(UAV_Agent):
    __UAV_MULTI_TARGET_AGENT_DEFAULT_ARGS = {
        "JBalanceFactor": .9,
        "deltaDisForMinDisThreshold": 10.,
    }

    def __init__(self, initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, targetNum, deltaTime=1., height=1., predictorCls=None,
                 predictorComputationArgs=None):
        super().__init__(initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                         optimizerInitArgs, optimizerComputationArgs, deltaTime, height)

        optimizerInitArgs["maxConstraint"].insert(0, float(targetNum) - 0.05)
        optimizerInitArgs["minConstraint"].insert(0, float(0.))
        optimizerInitArgs["optimizerLearningDimNum"] += 1
        optimizerComputationArgs["borders"].insert(0, 1)

        self.optimizer = optimizerCls(n=optimizerInitArgs["n"],
                                      dimNum=optimizerInitArgs["optimizerLearningDimNum"],
                                      maxConstraint=optimizerInitArgs["maxConstraint"],
                                      minConstraint=optimizerInitArgs["minConstraint"],
                                      evalVars=self.evalVars,
                                      otimizeWay=optimizerInitArgs["optimizeWay"],
                                      needEpochTimes=optimizerInitArgs["needEpochTimes"],
                                      ECArgs=optimizerComputationArgs,
                                      otherTerminalHandler=self.optimizerTerminalHandler)

        self.targetNum = targetNum
        self.predictorCls = predictorCls
        self.predictorComputationArgs = predictorComputationArgs
        self.trackingTargetIndex = 0
        self.testTrackingTargetIndex = 0

        # 如果已经有了JBalanceFactor，由于只添加不存在的，所以使用用户输入的
        # 如果没有，就使用默认的
        self.agentArgs.update(newDict=self.__UAV_MULTI_TARGET_AGENT_DEFAULT_ARGS, onlyAddNotExists=True)

    def recvMeg(self, **kwargs):
        self.agentCrowd = kwargs["agentCrowd"]
        self.selfIndex = kwargs["selfIndex"]
        self.targetPositionList = self.predictTargetMoving(kwargs["targetPositionList"])
        self.numOfTrackingUAVForTargetList = kwargs["numOfTrackingUAVForTargetList"]

        self.optimizer.ECDynOptHyperMutation_ECArgsDictValueController["performanceThreshold"] = 1 / (
                calcDistance(self.positionState, self.targetPositionList[self.trackingTargetIndex][0]) *
                self.optimizer.ECArgsDictValueController["fittingMinDenominator"])

        if len(self.agentCrowd["predictVelocityList"]) > 0:
            if len(self.agentCrowd["predictVelocityList"][0]) < self.optimizer.dimNum:
                for i in range(len(self.agentCrowd["predictVelocityList"])):
                    predictVelocityListForItemList = self.agentCrowd["predictVelocityList"][i].tolist()
                    predictVelocityListForItemList.insert(0, 0.)
                    self.agentCrowd["predictVelocityList"][i] = np.array(predictVelocityListForItemList)

    # def optimization(self):
    #     super().optimization()

    def moving(self):
        self.trackingTargetIndex = int(self.predictVelocityList[0])
        startIndex, endIndex = self.getVelocityFromPredictVelocityList(
            self.agentArgs["usePredictVelocityLen"] - self.remainMoving)
        self.positionState = calcMovingForUAV(self.positionState, self.predictVelocityList[startIndex: endIndex],
                                              self.deltaTime)

    def predictTargetMoving(self, targetPositionList):
        return targetPositionList

    def evalVars(self, chromosome):
        # return self.agentArgs["JTaskFactor"] * self.evalVars_JTask(chromosome, self.predictVelocityLen) + \
        #        self.agentArgs["JConFactor"] * self.evalVars_JConsume(chromosome,self.predictVelocityLen) + \
        #        self.agentArgs["JColFactor"] * self.evalVars_JCollision(chromosome, self.predictVelocityLen) + \
        #        self.agentArgs["JComFactor"] * self.evalVars_JCommunication(chromosome, self.predictVelocityLen)
        return self.agentArgs["JBalanceFactor"] * self.evalVars_JBalance(chromosome, self.predictVelocityLen) + \
               self.agentArgs["JTaskFactor"] * self.evalVars_JTask(chromosome, self.predictVelocityLen) + \
               self.agentArgs["JConFactor"] * self.evalVars_JConsume(chromosome, self.predictVelocityLen) + \
               self.agentArgs["JColFactor"] * self.evalVars_JCollision(chromosome, self.predictVelocityLen)

    def getVelocityFromPredictVelocityList(self, velocityIndex):
        velocityLen = self.velocity.size
        return int((velocityIndex) * velocityLen) + 1, int((velocityIndex + 1) * velocityLen) + 1

    def evalVars_JBalance(self, chromosome, *args):
        if hasattr(self, "maxTrackingTargetIndexVar") is False:
            maxTrackingTargetIndexVarList = np.zeros(self.numOfTrackingUAVForTargetList.size)
            maxTrackingTargetIndexVarList[0] = len(self.agentCrowd)
            self.maxTrackingTargetIndexVar = np.var(maxTrackingTargetIndexVarList)

        self.testTrackingTargetIndex = int(chromosome[0])
        # if self.testTrackingTargetIndex == self.numOfTrackingUAVForTargetList.size:
        #     self.testTrackingTargetIndex = self.testTrackingTargetIndex - 1
        #     chromosome[0] = float(self.testTrackingTargetIndex)

        numOfTrackingUAVForTargetList = np.array(self.numOfTrackingUAVForTargetList)
        numOfTrackingUAVForTargetList[self.testTrackingTargetIndex] += 1.
        if numOfTrackingUAVForTargetList[self.trackingTargetIndex] >= 1.:
            numOfTrackingUAVForTargetList[self.trackingTargetIndex] -= 1.
        return np.var(numOfTrackingUAVForTargetList) / self.maxTrackingTargetIndexVar

    def evalVars_JTask(self, chromosome, *args):
        predictVelocityLen = args[0]
        JTaskList = np.zeros(predictVelocityLen)
        oldPositionList = [self.positionState]
        originDisFromTarget = 0.
        trackingTargetPosition = self.targetPositionList[self.testTrackingTargetIndex]

        for i in range(predictVelocityLen):
            startIndex, endIndex = self.getVelocityFromPredictVelocityList(i)
            newPositionState = calcMovingForUAV(oldPositionList[i], chromosome[startIndex: endIndex], self.deltaTime)
            targetPosition = trackingTargetPosition[i]
            JTaskList[i] = np.square(newPositionState[0] - targetPosition[0]) + np.square(
                newPositionState[1] - targetPosition[1])
            if i == 0:
                originDisFromTarget = np.square(self.positionState[0] - targetPosition[0]) + np.square(
                    self.positionState[1] - targetPosition[1])
            oldPositionList.append(newPositionState)
        JTaskMax = max(np.max(JTaskList), originDisFromTarget * 2.)
        JTaskVal = ((np.average(JTaskList)) / (JTaskMax))
        return JTaskVal

    def evalVars_JCollision(self, chromosome, *args):
        def JCollisionMeasureFunc(distanceFromItem):
            return 0.5 - np.tanh(8 * (distanceFromItem - self.agentArgs["minDistanceThreshold"] - 0.5 * self.agentArgs[
                "deltaDisForMinDisThreshold"]) / self.agentArgs["deltaDisForMinDisThreshold"])

        predictVelocityLen = args[0]
        JCollisionList = []

        for index, item in enumerate(self.agentCrowd["positionState"]):
            if index != self.selfIndex:
                oldPositionList = self.positionState
                oldPositionListForItem = item
                chromosomeForItem = self.agentCrowd["predictVelocityList"][index]

                for i in range(predictVelocityLen):
                    startIndex, endIndex = self.getVelocityFromPredictVelocityList(i)
                    newPositionState = calcMovingForUAV(oldPositionList, chromosome[startIndex: endIndex],
                                                        self.deltaTime)
                    newPositionStateForItem = calcMovingForUAV(oldPositionListForItem,
                                                               chromosomeForItem[startIndex: endIndex],
                                                               self.deltaTime)
                    distanceFromItem = calcDistance(newPositionState[0: 2], newPositionStateForItem[0: 2])
                    JCollisionList.append(JCollisionMeasureFunc(distanceFromItem))

                    oldPositionList = newPositionState
                    oldPositionListForItem = newPositionStateForItem

        JCollisionVal = np.average(np.array(JCollisionList))

        return JCollisionVal
