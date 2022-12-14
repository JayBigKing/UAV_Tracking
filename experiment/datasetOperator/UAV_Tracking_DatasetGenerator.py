#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_Tracking_DatasetGenerator.py
@Author  : jay.zhu
@Time    : 2022/11/26 17:39
"""
import random
import time
import json
import numpy as np
from Jay_Tool.LogTool import myLogger
from MAS.Agents.UAV_Agent import UAV_TargetAgent

class UAV_Tracking_DatasetGenerator:
    def __init__(self):
        pass

    def setGeneratorArgs(self, agentNum, targetNum, movingTimes,
                         agentPositionDimension=3, targetPositionDimension=3,
                         agentInitPosRange=None, targetInitPosRange=None,
                         agentInitPosPresetVec=None, targetInitPosPresetVec=None,
                         targetMovingWay=None, targetMovingWayVec=None):
        self.agentNum = agentNum
        self.targetNum = targetNum
        self.movingTimes = movingTimes
        self.agentPositionDimension = agentPositionDimension
        self.targetPositionDimension = targetPositionDimension
        self.agentInitPosRange = np.array(agentInitPosRange) if agentInitPosRange is not None else None
        self.targetInitPosRange = np.array(targetInitPosRange) if targetInitPosRange is not None else None
        self.agentInitPosPresetVec = np.array(agentInitPosPresetVec) if agentInitPosPresetVec is not None else None
        self.targetInitPosPresetVec = np.array(targetInitPosPresetVec) if targetInitPosPresetVec is not None else None
        self.targetMovingWay = targetMovingWay
        self.targetMovingWayVec = targetMovingWayVec

        if self.agentInitPosPresetVec is None:
            for i in range(self.agentPositionDimension):
                self.agentInitPosRange[i, 0], self.agentInitPosRange[i, 1] = np.min(self.agentInitPosRange[i, :]), np.max(
                    self.agentInitPosRange[i, :])

        if self.targetInitPosPresetVec is None:
            for i in range(self.targetPositionDimension):
                self.targetInitPosRange[i, 0], self.targetInitPosRange[i, 1] = np.min(
                    self.targetInitPosRange[i, :]), np.max(self.targetInitPosRange[i, :])

    def checkGeneratorArgs(self):
        try:
            if self.agentNum <= 0:
                raise ValueError("agentNum should be greater than 0")
            elif self.targetNum <= 0:
                raise ValueError("targetNum should be greater than 0")

            if self.agentInitPosRange is None and self.agentInitPosPresetVec is None:
                raise ValueError("There is no agent init position range or preset vec")
            elif self.agentInitPosPresetVec is not None:
                if len(self.agentInitPosPresetVec) != self.agentNum:
                    raise ValueError("The len of agentInitPosPresetVec should be equal to agentNum")

            if self.targetInitPosRange is None and self.targetInitPosPresetVec is None:
                raise ValueError("There is no target init position range or preset vec")
            elif self.targetInitPosPresetVec is not None:
                if len(self.targetInitPosPresetVec) != self.targetNum:
                    raise ValueError("The len of targetInitPosPresetVec should be equal to targetNum")

            if self.targetMovingWay is None and self.targetMovingWayVec is None:
                raise ValueError("Should set targetMovingWay or a vec of targetMovingWay")
            elif self.targetMovingWayVec is not None:
                if len(self.targetMovingWayVec) != self.targetNum:
                    raise ValueError("The len of targetMovingWayVec should be equal to targetNum")

            return True

        except ValueError as e:
            myLogger.myLogger_Logger().error(repr(e))
            return False

    def generateDataset(self, agentNum, targetNum, movingTimes,
                        agentPositionDimension=3, targetPositionDimension=3,
                        agentInitPosRange=None, targetInitPosRange=None,
                        agentInitPosPresetVec=None, targetInitPosPresetVec=None,
                        targetMovingWay=None, targetMovingWayVec=None,
                        fileName=None, storePath=None):
        self.setGeneratorArgs(agentNum, targetNum, movingTimes,
                              agentPositionDimension, targetPositionDimension,
                              agentInitPosRange, targetInitPosRange,
                              agentInitPosPresetVec, targetInitPosPresetVec,
                              targetMovingWay, targetMovingWayVec)
        try:
            if self.checkGeneratorArgs() is False:
                raise ValueError("Generator Args check fail")
        except ValueError as e:
            if myLogger.myLogger_Logger() is not None:
                myLogger.myLogger_Logger().error(repr(e))
            else:
                print(repr(e))
            return

        if self.agentInitPosPresetVec is None:
            agentInitPositionVec = [
                [random.uniform(self.agentInitPosRange[j, 0], self.agentInitPosRange[j, 1]) for j in
                 range(self.agentPositionDimension)]
                for i in range(self.agentNum)
            ]
        else:
            agentInitPositionVec = self.agentInitPosPresetVec

        if self.targetInitPosPresetVec is None:
            targetInitPositionVec = [
                [random.uniform(self.targetInitPosRange[j, 0], self.targetInitPosRange[j, 1]) for j in
                 range(self.targetPositionDimension)]
                for i in range(self.targetNum)
            ]
        else:
            targetInitPositionVec = self.targetInitPosPresetVec

        if self.targetMovingWayVec is None:
            targetVec = [UAV_TargetAgent.UAV_TargetAgent(
                initPositionState=[targetInitPositionVec[i][j] for j in range(self.targetPositionDimension)],
                linearVelocityRange=[0, 0],
                angularVelocityRange=[0, 0],
                movingFuncRegister=self.targetMovingWay) for i in range(self.targetNum)]
        else:
            targetVec = [UAV_TargetAgent.UAV_TargetAgent(
                initPositionState=[targetInitPositionVec[i][j] for j in range(self.targetPositionDimension)],
                linearVelocityRange=[0, 0],
                angularVelocityRange=[0, 0],
                movingFuncRegister=self.targetMovingWayVec[i]) for i in range(self.targetNum)]

        targetTrajectories = [{
            "targetPositionVec": [],
            "targetVelocityVec": []
        } for i in range(self.targetNum)]

        for i in range(self.movingTimes):
            for j in range(self.targetNum):
                targetVec[j].update()
                targetTrajectories[j]["targetPositionVec"].append(targetVec[j].positionState.tolist())
                targetTrajectories[j]["targetVelocityVec"].append(targetVec[j].velocity.tolist())

        self.saveToJson(
            self.packJsonData(agentNum, targetNum, agentInitPositionVec, targetInitPositionVec, targetTrajectories),
            fileName, storePath)

    @staticmethod
    def packJsonData(agentsNum, targetNum, agentInitPositionVec, targetInitPositionVec, targetTrajectories):
        jsonPack = {
            "agentNum": agentsNum,
            "targetNum": targetNum,
            "agentInitPositionVec": agentInitPositionVec,
            "targetInitPositionVec": targetInitPositionVec,
            "targetTrajectories": targetTrajectories
        }
        return jsonPack

    def saveToJson(self, jsonPack, fileName=None, storePath=None):
        if fileName is None:
            fileName = time.strftime("agentTrackingDataset_%Y%m%d_%H%M%S.json", time.localtime())
        if storePath is None:
            storePath = "./"

        storeFileNameWithPath = "%s%s" % (storePath, fileName)
        with open(storeFileNameWithPath, "w") as f:
            json.dump(jsonPack, f)


# myLogger.myLogger_Init()
# dg = UAV_Tracking_DatasetGenerator()
# dg.generateDataset(2, 2,
#                    movingTimes=10,
#                    agentInitPosRange=[[0., 0.], [2., 2.], [3., 3.]],
#                    targetInitPosRange=[[4., 4.], [5., 5.], [0., 0.]],
#                    targetMovingWay="movingAsSin")
