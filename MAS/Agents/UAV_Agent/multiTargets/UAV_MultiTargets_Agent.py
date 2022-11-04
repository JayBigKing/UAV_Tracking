#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : UAV_MultiTargets_Agent.py
@Author  : jay.zhu
@Time    : 2022/11/4 20:03
"""
import numpy as np

from MAS.Agents.UAV_Agent.UAV_Common import calcMovingForUAV, calcDistance
from MAS.Agents.UAV_Agent.UAV_Agent import UAV_Agent

class UAV_MultiTargets_Agent(UAV_Agent):
    __UAV_MULTITARGETS_AGENT_DEFAULT_ARGS = {
        "JBalanceFactor": .9,
    }

    def __init__(self, initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, targetNum, deltaTime=1., height=1., predictorCls=None,
                 predictorComputationArgs=None):
        super().__init__(initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                         optimizerInitArgs, optimizerComputationArgs, deltaTime, height)

        optimizerInitArgs["maxConstraint"].insert(0, float(targetNum + 1))
        optimizerInitArgs["minConstraint"].insert(0, float(0.))
        optimizerInitArgs["optimizerLearningDimNum"] += 1

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

        # 如果已经有了JBalanceFactor，由于只添加不存在的，所以使用用户输入的
        # 如果没有，就使用默认的
        self.agentArgs.update(newDict=self.__UAV_MULTITARGETS_AGENT_DEFAULT_ARGS, onlyAddNotExists=True)

    def recvMeg(self, **kwargs):
        self.agentCrowd = kwargs["agentCrowd"]
        self.selfIndex = kwargs["selfIndex"]
        self.targetPositionList = self.predictTargetMoving(kwargs["targetPositionList"])
        self.numOfTrackingUAVForTargetList = kwargs["numOfTrackingUAVForTargetList"]

        self.optimizer.ECDynOptHyperMutation_ECArgsDictValueController["performanceThreshold"] = 1 / (
                calcDistance(self.positionState, self.targetPositionList[self.trackingTargetIndex][0]) *
                self.optimizer.ECArgsDictValueController["fittingMinDenominator"])

    def moving(self):
        self.trackingTargetNum = int(self.predictVelocityList[0])
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
        return self.agentArgs["JTaskFactor"] * self.evalVars_JTask(chromosome,
                                                                   self.predictVelocityLen) + \
               self.agentArgs["JConFactor"] * self.evalVars_JConsume(chromosome, self.predictVelocityLen) + \
               self.agentArgs["JBalanceFactor"] * self.evalVars_JBalance(chromosome, self.predictVelocityLen)

    def getVelocityFromPredictVelocityList(self, velocityIndex):
        velocityLen = self.velocity.size
        return int((velocityIndex) * velocityLen) + 1, int((velocityIndex + 1) * velocityLen) + 1

    def evalVars_JBalance(self, chromosome, *args):
        self.trackingTargetNum = int(chromosome[0])
        numOfTrackingUAVForTargetList = np.array(self.numOfTrackingUAVForTargetList)
        numOfTrackingUAVForTargetList[self.trackingTargetNum] += 1.
        return np.var(numOfTrackingUAVForTargetList)

    def evalVars_JTask(self, chromosome, *args):
        predictVelocityLen = args[0]
        JTaskList = np.zeros(predictVelocityLen)
        oldPositionList = [self.positionState]
        originDisFromTarget = 0.
        trackingTargetPosition = self.targetPositionList[self.trackingTargetNum]

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
