#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : UAV_MultiTargets_MAS_Base.py
@Author  : jay.zhu
@Time    : 2022/11/3 21:02
"""
import numpy as np
from MAS.MultiAgentSystem.UAV_MAS.UAV_MAS_Base import UAV_MAS_Base
from algorithmTool.filterTool.ExtendedKalmanFilter import ExtendedKalmanFilter
from EC.EC_Common import ArgsDictValueController


class UAV_MultiTargets_MAS_Base(UAV_MAS_Base):
    __UAV_MULTITARGETS_MAS_BASE_DEFAULT_ARGS = {
        "linearVelocityConsumeFactor": 1.,
        "angularVelocityConsumeFactor": 1.
    }

    def __init__(self, agents, masArgs, targetNum, terminalHandler=None, statRegisters=None, deltaTime=1.):
        super().__init__(agents, masArgs, terminalHandler, statRegisters)
        self.targetNum = targetNum
        self.deltaTime = deltaTime

        self.targetPositionList = None
        self.numOfTrackingUAVForTargetList = np.zeros(self.targetNum)

        self.nowRunningGen = 0
        self.UAV_MultiTargets_MAS_Base_Args = ArgsDictValueController(masArgs,
                                                                      self.__UAV_MULTITARGETS_MAS_BASE_DEFAULT_ARGS,
                                                                      onlyUseDefaultKey=True)

    def updateAgentState(self):
        super().updateAgentState()
        self.updateNumOfTrackingUAVForTargets()

    def updateNumOfTrackingUAVForTargets(self):
        for index in range(self.numOfTrackingUAVForTargetList.size):
            self.numOfTrackingUAVForTargetList[index] = 0.
        for item in self.agents:
            trackingTargetIndex = item.trackingTargetIndex
            self.numOfTrackingUAVForTargetList[trackingTargetIndex] += 1.

    def optimizationPreProcess(self):
        agentsPositionState = []
        agentsVelocity = []
        predictVelocityList = []

        for item in self.agents:
            agentsPositionState.append(item.positionState)
            agentsVelocity.append(item.velocity)
            predictVelocityList.append(item.predictVelocityList)

        self.agentCrowd = {
            "positionState": agentsPositionState,
            "velocity": agentsVelocity,
            "predictVelocityList": predictVelocityList
        }

        for index, item in enumerate(self.agents):
            try:
                if self.targetPositionList is not None:
                    item.recvMeg(agentCrowd=self.agentCrowd, selfIndex=index,
                                 targetPositionList=self.targetPositionList,
                                 numOfTrackingUAVForTargetList=self.numOfTrackingUAVForTargetList)
                else:
                    raise NotImplementedError("TargetPositionList is not implement\r\n"
                                              "Please implement the variable that contains each targets position"
                                              "or trajectory")
            except NotImplementedError as e:
                print(repr(e))

    '''
    following is stat function
    '''

    """
    @brief: record the num of uav tracking the target for each target.
    """

    def UAV_MultiTargets_MAS_Stat_numOfTrackingUAVForTarget(self, **kwargs):
        if hasattr(self, "numOfTrackingUAVForTargetStat") is False:
            self.numOfTrackingUAVForTargetStat = []
            for item in self.numOfTrackingUAVForTargetList:
                self.numOfTrackingUAVForTargetStat.append([np.array([float(self.nowRunningGen), item])])
        else:
            for index, item in enumerate(self.numOfTrackingUAVForTargetList):
                self.numOfTrackingUAVForTargetStat[index].append(
                    np.array([float(self.nowRunningGen), item.trackingTargetIndex]))

    """
    @brief: record consume of each uav.
    """

    def UAV_MultiTargets_MAS_Stat_ConsumeOfEachUAV(self, **kwargs):
        def calcAgentConsume(agent):
            remainMoving = agent.remainMoving + 1
            startIndex, _ = agent.getVelocityFromPredictVelocityList(remainMoving)
            itemConsume = (agent.chromosome[startIndex] * self.UAV_MultiTargets_MAS_Base_Args[
                "linearVelocityConsumeFactor"] +
                           agent.chromosome[startIndex + 1] * self.UAV_MultiTargets_MAS_Base_Args[
                               "angularVelocityConsumeFactor"]) \
                          * self.deltaTime
            return itemConsume

        if hasattr(self, "consumeOfEachUAVStat") is False:
            self.ConsumeOfEachUAVStat = []
            for item in self.agents:
                self.ConsumeOfEachUAVStat.append([np.array([float(self.nowRunningGen), calcAgentConsume(item)])])
        else:
            for index, item in enumerate(self.agents):
                self.ConsumeOfEachUAVStat[index].append(np.array([float(self.nowRunningGen), calcAgentConsume(item)]))
