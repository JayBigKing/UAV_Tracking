#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : Agent_UAV_Base.py
@Author  : jay.zhu
@Time    : 2022/10/13 12:38
"""
import numpy as np
from MAS.Agents.Agent_WithStat_Base import Agent_WithStat_Base
from MAS.Agents.UAV_Agent.UAV_Common import calcMovingForUAV


class Agent_UAV_Base(Agent_WithStat_Base):
    def __init__(self, initPositionState, linearVelocityRange, angularVelocityRange, maxVariationOfLinearSpeed = 2., optimizer=None, deltaTime=1.):
        super().__init__(optimizer=optimizer, statRegisters=[self.coordinateRecord])
        self.positionState = np.array(initPositionState)  # 0:x轴，1:y轴，2:航向角
        self.linearVelocityRange = np.array(linearVelocityRange)
        self.linearSpeedChangeRange = np.array([-abs(maxVariationOfLinearSpeed), abs(maxVariationOfLinearSpeed)])
        self.angularVelocityRange = np.array(angularVelocityRange)
        if self.linearVelocityRange[0] > self.linearVelocityRange[1]:
            self.linearVelocityRange[0], self.linearVelocityRange[1] = self.linearVelocityRange[1], \
                                                                       self.linearVelocityRange[0]
        if self.angularVelocityRange[0] > self.angularVelocityRange[1]:
            self.angularVelocityRange[0], self.angularVelocityRange[1] = self.angularVelocityRange[1], self.angularVelocityRange[0]

        self.velocity = np.zeros(2)  # 索引1是线速度，索引2是角速度
        self.deltaTime = deltaTime
        self.coordinateVector = [[self.positionState[0], self.positionState[1]]]

    def updateInner(self):
        self.moving()
        # for item in self.statFuncReg:
        #     item(positionState=True)

    def moving(self):
        self.velocity[0], self.velocity[1] = self.optimizationResult[0][0], self.optimizationResult[0][1]
        self.positionState = calcMovingForUAV(self.positionState, self.velocity, self.deltaTime)

    def coordinateRecord(self, **kwargs):
        # if kwargs.get("positionState"):
        self.coordinateVector.append([self.positionState[0], self.positionState[1]])