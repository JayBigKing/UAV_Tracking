#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_MAS_Base.py
@Author  : jay.zhu
@Time    : 2022/11/3 21:02
"""
import numpy as np
from MAS.MultiAgentSystem.UAV_MAS.UAV_MAS_Base import UAV_MAS_Base
from optimization.common.ArgsDictValueController import ArgsDictValueController
from MAS.Agents.UAV_Agent.UAV_Common import calcDistance


class UAV_MultiTarget_MAS_Base(UAV_MAS_Base):
    __UAV_MULTI_TARGET_MAS_BASE_DEFAULT_ARGS = {
        "linearVelocityConsumeFactor": 1.,
        "angularVelocityConsumeFactor": 1.,
        "effectiveTimeDisThreshold": 20.,
    }

    def __init__(self, agents, masArgs, targetNum, terminalHandler=None, statRegisters=None, deltaTime=1.):
        self.__UAV_MULTI_TARGET_MAS_BASE_DEFAULT_STAT_FUNC_DICT = {
            "recordNumOfTrackingUAVForTarget": self.UAV_MultiTargets_MAS_Stat_numOfTrackingUAVForTarget,
            "recordConsumeOfEachUAV": self.UAV_MultiTargets_MAS_Stat_ConsumeOfEachUAV,
            "recordDisBetweenTargetAndUAV": self.UAV_MultiTargets_MAS_Stat_disBetweenTargetAndUAV,
            "recordEffectiveTime": self.UAV_MultiTargets_MAS_Stat_EffectiveTime,
            "recordEffectiveTimeForTarget":self.UAV_MultiTargets_MAS_Stat_EffectiveTimeForTarget,
            "recordTrackTargetID": self.UAV_MultiTargets_MAS_Stat_recordTrackTargetID,
            "recordDisBetweenCloseTar_UAV": self.UAV_MultiTargets_MAS_Stat_disBetweenCloseTar_UAV

        }

        if statRegisters is None:
            statRegisters = [self.UAV_MultiTargets_MAS_Stat_numOfTrackingUAVForTarget,
                             "recordDisOfUAVsForVisualize",
                             "recordAlertDisOfUAVsForVisualize",
                             self.UAV_MultiTargets_MAS_Stat_disBetweenTargetAndUAV]
        super().__init__(agents, masArgs, terminalHandler, statRegisters,
                         self.__UAV_MULTI_TARGET_MAS_BASE_DEFAULT_STAT_FUNC_DICT)
        self.targetNum = targetNum
        self.deltaTime = deltaTime

        self.targetPositionList = None
        self.numOfTrackingUAVForTargetList = np.zeros(self.targetNum)

        self.UAV_MultiTargets_MAS_Base_Args = ArgsDictValueController(masArgs,
                                                                      self.__UAV_MULTI_TARGET_MAS_BASE_DEFAULT_ARGS,
                                                                      onlyUseDefaultKey=True)
        self.realTargetPosition = np.zeros((2, targetNum))

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
            predictVelocityList.append(np.array([0. for i in range(len(item.predictVelocityList))]))

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

    def recvFromEnv(self, **kwargs):
        for i in range(self.targetNum):
            self.realTargetPosition[:, i] = np.array([kwargs["targetPosition"][i][0],
                                                      kwargs["targetPosition"][i][1]])

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
                    np.array([float(self.nowRunningGen), item]))

    """
    @brief: record consume of each uav.
    """

    def UAV_MultiTargets_MAS_Stat_ConsumeOfEachUAV(self, **kwargs):
        def calcAgentConsume(agent):
            remainMoving = agent.remainMoving + 1
            startIndex, _ = agent.getVelocityFromPredictVelocityList(
                agent.agentArgs["usePredictVelocityLen"] - remainMoving)
            itemConsume = (agent.predictVelocityList[startIndex] * self.UAV_MultiTargets_MAS_Base_Args[
                "linearVelocityConsumeFactor"] +
                           abs(agent.predictVelocityList[startIndex + 1]) * self.UAV_MultiTargets_MAS_Base_Args[
                               "angularVelocityConsumeFactor"]) \
                          * self.deltaTime
            return itemConsume

        if hasattr(self, "consumeOfEachUAVStat") is False:
            self.consumeOfEachUAVStat = []
            for item in self.agents:
                self.consumeOfEachUAVStat.append([np.array([float(self.nowRunningGen), calcAgentConsume(item)])])
        else:
            for index, item in enumerate(self.agents):
                self.consumeOfEachUAVStat[index].append(np.array([float(self.nowRunningGen), calcAgentConsume(item)]))

    """
    @brief: record the distance between target tracked and uav
    """

    def UAV_MultiTargets_MAS_Stat_disBetweenTargetAndUAV(self, **kwargs):
        def calcDistanceBetweenTargetAndUAV(agent):
            trackingTargetIndex = agent.trackingTargetIndex
            return calcDistance(agent.positionState[0: 2], self.realTargetPosition[:, trackingTargetIndex])

        if hasattr(self, "disBetweenTargetAndUAVStat") is False:
            self.disBetweenTargetAndUAVStat = []
            for item in self.agents:
                self.disBetweenTargetAndUAVStat.append(
                    [np.array([float(self.nowRunningGen), calcDistanceBetweenTargetAndUAV(item)])])
        else:
            for index, item in enumerate(self.agents):
                self.disBetweenTargetAndUAVStat[index].append(
                    np.array([float(self.nowRunningGen), calcDistanceBetweenTargetAndUAV(item)]))

        if self.UAV_MultiTargets_MAS_Stat_EffectiveTime in self.statFuncReg:
            self.UAV_MultiTargets_MAS_Stat_EffectiveTime(run=True)
        if self.UAV_MultiTargets_MAS_Stat_EffectiveTimeForTarget in self.statFuncReg:
            self.UAV_MultiTargets_MAS_Stat_EffectiveTimeForTarget(run=True)

    def UAV_MultiTargets_MAS_Stat_EffectiveTime(self, **kwargs):
        if kwargs.get("run"):
            if hasattr(self, "effectiveTimeStat") is False:
                self.effectiveTimeStat = [[] for item in self.agents]
                self.__effectiveTimeMiddleRecode = [0. for item in self.agents]
                self.__effectiveTimeFlag = [False for item in self.agents]
            else:
                for index, item in enumerate(self.disBetweenTargetAndUAVStat):
                    if item[-1][1] <= self.__UAV_MULTI_TARGET_MAS_BASE_DEFAULT_ARGS["effectiveTimeDisThreshold"]:
                        if self.__effectiveTimeFlag[index] is False:
                            self.__effectiveTimeFlag[index] = True
                            self.__effectiveTimeMiddleRecode[index] = self.deltaTime
                        else:
                            self.__effectiveTimeMiddleRecode[index] += self.deltaTime
                    else:
                        if self.__effectiveTimeFlag[index] is True:
                            self.__effectiveTimeFlag[index] = False
                            self.effectiveTimeStat[index].append(self.__effectiveTimeMiddleRecode[index])

    def UAV_MultiTargets_MAS_Stat_EffectiveTimeForTarget(self, **kwargs):
        if kwargs.get("run"):
            if hasattr(self, "effectiveTimeForTargetStat") is False:
                self.effectiveTimeForTargetStat = [[] for item in range(self.targetNum)]
                self.effectivePointForTargetStat = [[] for item in range(self.targetNum)]
                self.__effectiveTimeMiddleRecodeForTarget = [0. for item in range(self.targetNum)]
                self.__effectiveTimeFlagForTarget = [False for item in range(self.targetNum)]
            else:
                for index in range(self.targetNum):
                    self.effectivePointForTargetStat[index].append(0)

                haveCalcTargetIndexSet = set()
                for index, item in enumerate(self.disBetweenTargetAndUAVStat):
                    trackingTargetIndex = int(self.agents[index].trackingTargetIndex)
                    if trackingTargetIndex not in haveCalcTargetIndexSet:
                        haveCalcTargetIndexSet.add(trackingTargetIndex)
                        if item[-1][1] <= self.__UAV_MULTI_TARGET_MAS_BASE_DEFAULT_ARGS["effectiveTimeDisThreshold"]:
                            self.effectivePointForTargetStat[trackingTargetIndex][-1] = 1
                            if self.__effectiveTimeFlagForTarget[trackingTargetIndex] is False:
                                self.__effectiveTimeFlagForTarget[trackingTargetIndex] = True
                                self.__effectiveTimeMiddleRecodeForTarget[trackingTargetIndex] = self.deltaTime
                            else:
                                self.__effectiveTimeMiddleRecodeForTarget[trackingTargetIndex] += self.deltaTime
                        else:
                            if self.__effectiveTimeFlagForTarget[trackingTargetIndex] is True:
                                self.__effectiveTimeFlagForTarget[trackingTargetIndex] = False
                                self.effectiveTimeForTargetStat[trackingTargetIndex].append(
                                    self.__effectiveTimeMiddleRecodeForTarget[trackingTargetIndex])

    """
    @brief: recode each uav's track target ID
    """

    def UAV_MultiTargets_MAS_Stat_recordTrackTargetID(self, **kwargs):
        if hasattr(self, 'trackTargetIDStat') is False:
            self.trackTargetIDStat = [[] for item in self.agents]

        for index, item in enumerate(self.agents):
            self.trackTargetIDStat[index].append([float(self.nowRunningGen), item.trackingTargetIndex])

    """
    @brief: record the distance between closest target tracked and uav
    """

    def UAV_MultiTargets_MAS_Stat_disBetweenCloseTar_UAV(self, **kwargs):
        def calcDistanceBetweenCloseTargetAndUAV(agent):
            return min([calcDistance(agent.positionState[0: 2], self.realTargetPosition[:, index]) for index in
                        range(self.targetNum)])

        if hasattr(self, "disBetweenCloseTarUAVStat") is False:
            self.disBetweenCloseTarUAVStat = []
            for item in self.agents:
                self.disBetweenCloseTarUAVStat.append(
                    [np.array([float(self.nowRunningGen), calcDistanceBetweenCloseTargetAndUAV(item)])])
        else:
            for index, item in enumerate(self.agents):
                self.disBetweenCloseTarUAVStat[index].append(
                    np.array([float(self.nowRunningGen), calcDistanceBetweenCloseTargetAndUAV(item)]))
