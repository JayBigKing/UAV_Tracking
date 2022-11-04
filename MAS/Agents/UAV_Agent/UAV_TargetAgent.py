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
    def __init__(self, initPositionState, linearVelocityRange, angularVelocityRange, maxVariationOfLinearSpeed=2.,
                 movingFuncRegister="movingStraightly",
                 deltaTime=1.):
        super().__init__(initPositionState, linearVelocityRange, angularVelocityRange, maxVariationOfLinearSpeed, None,
                         deltaTime)
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
        self.positionState = calcMovingForUAV(self.positionState, self.velocity, self.deltaTime)

    def movingAsSin(self, *args):
        def getMoivingAngleForSin(lastX, xMovingStepLength, yPeakValue, w=0.2):
            return np.rad2deg(np.arctan2(yPeakValue * (np.sin(w * (lastX - xMovingStepLength)) - np.sin(w * lastX)),
                                         xMovingStepLength))

        if hasattr(self, "movingAsSinInitFlag") is False:
            self.movingAsSinInitFlag = True
            self.X_MOVING_STEP_LENGTH = 4.
            self.Y_PEAK_VALUE = 10.
            self.positionState[2] = getMoivingAngleForSin(self.positionState[0], self.X_MOVING_STEP_LENGTH,
                                                          self.Y_PEAK_VALUE)
            self.velocity[0], self.velocity[1] = 0., 0.


        self.velocity[0] = self.X_MOVING_STEP_LENGTH / np.cos(np.deg2rad(self.positionState[2]))
        self.velocity[1] = (getMoivingAngleForSin(self.positionState[0] + self.X_MOVING_STEP_LENGTH,
                                                    self.X_MOVING_STEP_LENGTH, self.Y_PEAK_VALUE) -
                                                    self.positionState[2]) / self.deltaTime

    def randMoving(self, *args):
        newLinearVelocity = self.velocity[0] + np.random.uniform(self.linearSpeedChangeRange[0],self.linearSpeedChangeRange[1])
        if abs(newLinearVelocity) < self.linearVelocityRange[0]:
            if newLinearVelocity < 0:
                self.velocity[0] = -self.linearVelocityRange[0]
            else:
                self.velocity[0] = self.linearVelocityRange[0]
        elif abs(newLinearVelocity) > self.linearVelocityRange[1]:
            if newLinearVelocity < 0:
                self.velocity[0] = -self.linearVelocityRange[1]
            else:
                self.velocity[0] = self.linearVelocityRange[1]
        else:
            self.velocity[0] = newLinearVelocity

        self.velocity[1] = np.random.uniform(self.angularVelocityRange[0], self.angularVelocityRange[1])

    def movingStraightly(self, *args):
        ANGLE_THAT_TAN_IS_2 = 63.43495
        VELOCITY_THAT_MUL_BY_SIN_IS_2 = 2.236068
        self.positionState[2] = ANGLE_THAT_TAN_IS_2
        self.velocity[0] = VELOCITY_THAT_MUL_BY_SIN_IS_2
        self.velocity[1] = 0.
