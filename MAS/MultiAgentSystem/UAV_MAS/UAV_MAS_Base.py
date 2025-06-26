#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : UAV_MAS_Base.py
@Author  : jay.zhu
@Time    : 2022/10/14 13:58
"""
import numpy as np

from optimization.common.ArgsDictValueController import ArgsDictValueController
from MAS.MultiAgentSystem.MAS_WithStat_Base import MAS_WithStat_Base
from MAS.Agents.UAV_Agent.UAV_Common import calcDistance


class UAV_MAS_Base(MAS_WithStat_Base):
    __UAV_MAS_BASE_DEFAULT_ARGS = {
        "lowerBoundOfUAVDis": 10.,
        "upperBoundOfUAVDis": 50.
    }

    def __init__(self, agents, masArgs, terminalHandler=None, statRegisters=None, statFuncDict=None):
        self.__UAV_MAS_BASE_DEFAULT_STAT_FUNC_DICT = {
            "recordDisOfUAVs": self._UAV_MAS_Stat_recordDisOfUAVs,
            "recordDisOfUAVsForVisualize": self._UAV_MAS_Stat_recordDisOfUAVsForVisualize,
            "recordAlertDisOfUAVsForVisualize": self._UAV_MAS_Stat_recordAlertDisOfUAVsForVisualize,
            "recordFitness":self._UAV_MAS_Stat_recordFitness
        }

        if statRegisters is None:
            statRegisters = [item for item in self.__UAV_MAS_BASE_DEFAULT_STAT_FUNC_DICT]

        if statFuncDict is not None:
            self.__UAV_MAS_BASE_DEFAULT_STAT_FUNC_DICT.update(statFuncDict)

        super().__init__(agents, masArgs, terminalHandler, statRegisters, self.__UAV_MAS_BASE_DEFAULT_STAT_FUNC_DICT)
        self.UAV_MAS_Base_Args = ArgsDictValueController(masArgs,
                                                         self.__UAV_MAS_BASE_DEFAULT_ARGS,
                                                         onlyUseDefaultKey=True)

        self.lastAgentOptimizationRes = []
        self.nowRunningGen = 0

    def update(self):
        super().update()
        self.nowRunningGen += 1

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
            item.recvMeg(agentCrowd=self.agentCrowd, selfIndex=index, targetPosition=self.targetPosition)

    def recvFromEnv(self, **kwargs):
        self.targetPosition = kwargs["targetPosition"]

    '''
    following is stat function
    '''
    """
    @brief: using a matrix that record that each one's distance from others.
    """

    def _UAV_MAS_Stat_recordDisOfUAVsInner(self, **kwargs):
        agentLen = len(self.agents)
        if hasattr(self, 'UAVDisMatrix') is False:
            self.UAVDisMatrix = np.zeros((agentLen, agentLen))
        if hasattr(self, 'UAVDisStatMatrix') is False:
            self.UAVDisStatMatrix = np.zeros((agentLen, agentLen, 2))
            for i in range(agentLen):
                for j in range(agentLen):
                    if i != j:
                        self.UAVDisStatMatrix[i, j, 1] = 1e10
        if hasattr(self, '__recordDisOfUAVsInnerCalledDict') is False:
            self.__recordDisOfUAVsInnerCalledDict = dict()

        if hasattr(self, 'UAVAvgDisListStat') is False:
            self.UAVAvgDisListStat = list()

        if hasattr(self, 'UAVMinDisListStat') is False:
            self.UAVMinDisListStat = list()

        if self.__recordDisOfUAVsInnerCalledDict.get(self.nowRunningGen) is not None:
            return
        else:
            self.__recordDisOfUAVsInnerCalledDict[self.nowRunningGen] = True

        if len(self.UAVMinDisListStat) == self.nowRunningGen:
            UAVAvgDisThisEpoch = 0.
            UAVMinDisThisEpoch = 1e10
            for index, item in enumerate(self.agents):
                for j in range(agentLen):
                    if index != j:
                        if j < index:
                            self.UAVDisMatrix[index, j] = self.UAVDisMatrix[j, index]
                        else:
                            self.UAVDisMatrix[index, j] = calcDistance(item.positionState[0: 2],
                                                                       self.agents[j].positionState[0: 2])
                        UAVAvgDisThisEpoch += self.UAVDisMatrix[index, j]
                        UAVMinDisThisEpoch = min(UAVMinDisThisEpoch, self.UAVDisMatrix[index, j])
                        self.UAVDisStatMatrix[index, j, 0] += self.UAVDisMatrix[index, j]
                        self.UAVDisStatMatrix[index, j, 1] = min(self.UAVDisMatrix[index, j],
                                                                 self.UAVDisStatMatrix[index, j, 1])
            self.UAVAvgDisListStat.append([float(self.nowRunningGen), UAVAvgDisThisEpoch / (agentLen * (agentLen - 1))])
            self.UAVMinDisListStat.append([float(self.nowRunningGen), UAVMinDisThisEpoch])

    """
    @brief: print out distance matrix in every Gen
    """
    def _UAV_MAS_Stat_recordDisOfUAVs(self, **kwargs):
        self._UAV_MAS_Stat_recordDisOfUAVsInner()

        print('recordDisOfUAVs: %r' % self.UAVDisMatrix)

    """
    @brief: using a list to store distinct every uav pair`s in every Gen for show later 
    """
    def _UAV_MAS_Stat_recordDisOfUAVsForVisualize(self, **kwargs):
        uavDisDescriptionFmt = "UAV %d and %d"
        agentLen = len(self.agents)
        if hasattr(self, 'UAVDisVisualizeStat') is False:
            self.UAVDisVisualizeStat = {uavDisDescriptionFmt % (i, j): [] for i in range(agentLen) for j in
                                        range(i + 1, agentLen)}

            if self.UAV_MAS_Base_Args["lowerBoundOfUAVDis"] is not False:
                self.UAVDisVisualizeStat["lowerBoundOfUAVDis"] = []
            if self.UAV_MAS_Base_Args["upperBoundOfUAVDis"] is not False:
                self.UAVDisVisualizeStat["upperBoundOfUAVDis"] = []

        self._UAV_MAS_Stat_recordDisOfUAVsInner()
        for i in range(agentLen):
            for j in range(i + 1, agentLen):
                self.UAVDisVisualizeStat[uavDisDescriptionFmt % (i, j)].append(
                    np.array([float(self.nowRunningGen), self.UAVDisMatrix[i, j]]))

        if self.UAVDisVisualizeStat.get("lowerBoundOfUAVDis") is not None:
            self.UAVDisVisualizeStat["lowerBoundOfUAVDis"].append(
                np.array([float(self.nowRunningGen), self.UAV_MAS_Base_Args["lowerBoundOfUAVDis"]]))
        if self.UAVDisVisualizeStat.get("upperBoundOfUAVDis") is not None:
            self.UAVDisVisualizeStat["upperBoundOfUAVDis"].append(
                np.array([float(self.nowRunningGen), self.UAV_MAS_Base_Args["upperBoundOfUAVDis"]]))

    """
    @brief: record any one of distinct every uav pair`s distance is lower of higher than threshold
    """
    def _UAV_MAS_Stat_recordAlertDisOfUAVsForVisualize(self, **kwargs):
        agentLen = len(self.agents)
        if hasattr(self, 'UAVAlertDisVisualizeStat') is False:
            self.UAVAlertDisVisualizeStat = dict()

            if self.UAV_MAS_Base_Args["lowerBoundOfUAVDis"] is not False:
                self.UAVAlertDisVisualizeStat["lowerThanLowerBound"] = []
            if self.UAV_MAS_Base_Args["upperBoundOfUAVDis"] is not False:
                self.UAVAlertDisVisualizeStat["upperThanUpperBound"] = []

        self._UAV_MAS_Stat_recordDisOfUAVsInner()
        lowerThanLowerBoundFlag = False
        upperThanUpperBoundFlag = False
        for i in range(agentLen):
            for j in range(i + 1, agentLen):
                if self.UAVAlertDisVisualizeStat.get("lowerThanLowerBound") is not None:
                    if self.UAVDisMatrix[i, j] < self.UAV_MAS_Base_Args["lowerBoundOfUAVDis"]:
                        lowerThanLowerBoundFlag = True
                else:
                    lowerThanLowerBoundFlag = True

                if self.UAVAlertDisVisualizeStat.get("upperThanUpperBound") is not None:
                    if self.UAVDisMatrix[i, j] > self.UAV_MAS_Base_Args["upperBoundOfUAVDis"]:
                        upperThanUpperBoundFlag = True
                else:
                    upperThanUpperBoundFlag = True

                if lowerThanLowerBoundFlag is True and upperThanUpperBoundFlag is True:
                    break


        if self.UAVAlertDisVisualizeStat.get("lowerThanLowerBound") is not None:
            self.UAVAlertDisVisualizeStat["lowerThanLowerBound"].append(
                np.array([float(self.nowRunningGen), 1. if lowerThanLowerBoundFlag is True else 0.]))
        if self.UAVAlertDisVisualizeStat.get("upperThanUpperBound") is not None:
            self.UAVAlertDisVisualizeStat["upperThanUpperBound"].append(
                np.array([float(self.nowRunningGen), 1. if upperThanUpperBoundFlag is True else 0.]))


    """
    @brief: recode each epoch the avg agent's fitness 
    """
    def _UAV_MAS_Stat_recordFitness(self, **kwargs):
        if hasattr(self, 'fitnessStat') is False:
            self.fitnessStat = []

        self.fitnessStat.append([float(self.nowRunningGen), np.average([item.optimizationResult[2] for item in self.agents])])
        # print('fitness: %f' % self.fitnessStat[-1][1])





