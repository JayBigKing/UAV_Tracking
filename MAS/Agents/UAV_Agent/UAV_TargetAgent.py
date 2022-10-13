#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : UAV_TargetAgent.py
@Author  : jay.zhu
@Time    : 2022/10/13 13:05
"""
import numpy as np
from MAS.Agents.UAV_Agent.UAV_Common import calcMovingForUAV

from MAS.Agents.UAV_Agent.Agent_UAV_Base import Agent_UAV_Base


class UAV_TargetAgent(Agent_UAV_Base):
    def __init__(self, initPositionState, linearVelocityRange, angularVelocity, movingFuncRegister="movingStraightly",
                 deltaTime=1.):
        super().__init__(initPositionState, linearVelocityRange, angularVelocity, None, deltaTime)
        self.__UAV_TARGET_AGENT_DEFAULT_MOVING_FUNCTION = {
            "movingAsSin": self.movingAsSin,
            "randMoving": self.randMoving,
            "movingStraightly": self.movingStraightly,
        }

        if isinstance(movingFuncRegister, str):
            if self.__UAV_TARGET_AGENT_DEFAULT_MOVING_FUNCTION.get(movingFuncRegister):
                self.movingFuncHandler = self.__UAV_TARGET_AGENT_DEFAULT_MOVING_FUNCTION[movingFuncRegister]
            else:
                raise ValueError("no such default moving moving."
                                 "maybe you can use a moving moving created by yourself, "
                                 "which will be re-call in every target moving time.")
        else:
            self.movingFuncHandler = movingFuncRegister

    def moving(self):
        self.movingFuncHandler(self.positionState, self.velocity, self.deltaTime)

    def movingAsSin(self, *args):

        self.positionState[0] = self.positionState[0] + self.deltaTime
        self.positionState[1] = self.positionState[1] + 4.5 * np.sin(np.deg2rad(self.positionState[2] * self.deltaTime))
        self.positionState[2] = self.positionState[2] + 90 * self.deltaTime

    def randMoving(self, *args):
        self.velocity[0], self.velocity[1] = np.random.uniform(self.linearVelocityRange[0],
                                                               self.linearVelocityRange[1]), np.random.uniform(
                                                                 self.angularVelocity[0], self.angularVelocity[1])
        self.positionState = calcMovingForUAV(self.positionState, self.velocity, self.deltaTime)

    def movingStraightly(self, *args):
        self.positionState[0] = self.positionState[0] + self.deltaTime
        self.positionState[1] = self.positionState[1] + 2 * self.deltaTime
