#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_Dataset_TargetAgent.py
@Author  : jay.zhu
@Time    : 2022/11/26 17:54
"""
import numpy as np
from Jay_Tool.LogTool import myLogger
from MAS.Agents.UAV_Agent.UAV_TargetAgent import UAV_TargetAgent

class UAV_Dataset_TargetAgent(UAV_TargetAgent):
    def __init__(self, initPositionState, targetTrajectory, deltaTime=1.):
        self.targetTrajectory = targetTrajectory
        self.targetTrajectoryLength = len(self.targetTrajectory["targetPositionVec"])
        try:
            if len(self.targetTrajectory["targetVelocityVec"]) != self.targetTrajectoryLength:
                raise ValueError("Len of targetVelocityVec is expected to equal to targetTrajectoryLength's")
        except ValueError as e:
            myLogger.myLogger_Logger().error(repr(e))
            return

        self.Dataset_TargetAgent_InitPositionState = initPositionState
        self.movingCount = -1
        super().__init__(initPositionState, [0., 0.], [0., 0.])

    def moving(self):
        if self.movingCount < self.targetTrajectoryLength - 1:
            self.movingCount += 1

    def __getattribute__(self, item):
        if item == "positionState":
            if self.movingCount == -1:
                return np.array(self.Dataset_TargetAgent_InitPositionState[0: 2])
            else:
                return np.array(self.targetTrajectory["targetPositionVec"][self.movingCount])

        elif item == "velocity":
            if self.movingCount == -1:
                return np.array([0., 0.])
            else:
                return np.array(self.targetTrajectory["targetVelocityVec"][self.movingCount])
        else:
            return super().__getattribute__(item)


