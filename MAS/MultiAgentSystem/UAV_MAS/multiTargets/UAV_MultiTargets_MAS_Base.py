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

class UAV_MultiTargets_MAS_Base(UAV_MAS_Base):
    def __init__(self, agents, masArgs, targetNum, terminalHandler=None, statRegisters=None):
        super().__init__(agents, masArgs, terminalHandler, statRegisters)
        self.targetNum = targetNum
        self.numOfTrackingUAVForTargetList = np.zeros(self.targetNum)

        self.numOfTrackingUAVForTargetStat = []


    def updateAgentState(self):
        super().updateAgentState()
        self.updateNumOfTrackingUAVForTargets()

    def updateNumOfTrackingUAVForTargets(self):
        for item in self.numOfTrackingUAVForTargetList:
            item = 0.
        for item in self.agents:
            trackingTargetIndex = item.trackingTargetIndex
            self.numOfTrackingUAVForTargetList[trackingTargetIndex] += 1.


    '''
    following is stat function
    '''
    def UAV_MultiTargets_MAS_Stat_numOfTrackingUAVForTarget(self, **kwargs):
        self.numOfTrackingUAVForTargetStat.append(self.numOfTrackingUAVForTargetList)


