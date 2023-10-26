#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : UAV_Scene_Base.py
@Author  : jay.zhu
@Time    : 2022/10/14 14:35
"""
import time
import os
import inspect
import numpy as np
import pandas as pd
from Jay_Tool.LogTool import myLogger
from Scene.Scene_Base import Scene_Base
from Jay_Tool.visualizeTool.CoorDiagram import CoorDiagram
from dataStatistics.statFuncListGenerator import statFuncListGenerator
from optimization.common.ArgsDictValueController import ArgsDictValueController


class UAV_Scene_Base(Scene_Base):
    UAV_SCENE_BASE_DEFAULT_ARGS = {
        "storeStatDataName": "storeStatData",
        "saveFigNameSuffix": "pdf"
    }
    SCENE_AND_MAS_RECORD_MAP = {
        "UAV_SCENE_BASE_UAVDisVisualize": "recordDisBetweenTargetAndUAV",
        "UAV_SCENE_BASE_UAVAlertDisVisualize": "recordDisOfUAVsForVisualize",
        "UAV_SCENE_BASE_UAVAvgDisStore": "recordDisOfUAVsForVisualize",
        "UAV_SCENE_BASE_UAVMinDisStore": "recordDisOfUAVsForVisualize",
        "UAV_MULTI_TARGET_SCENE_BASE_UAVAlertDisStore": "UAV_MULTI_TARGET_SCENE_BASE_UAVAlertDisStore",
        "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore": "recordFitness"
    }
    initArgs = {"agent": {}, "target": {}, "MAS": {}}

    NARROW_FIGURE_SIZE = (6, 2.8)

    def __init__(self, agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                 MAS_Args, needRunningTime, targetNum=1, deltaTime=1., figureSavePath=None, statOutputRegisters=None,
                 statOutputDict=None, sceneArgs=None):
        self.agentsNum = agentsNum
        self.targetNum = targetNum
        self.deltaTime = deltaTime
        self.figureSavePath = figureSavePath
        self.csvSavePath = figureSavePath
        self.__UAV_Scene_Base_stat_output_dict = {
            "UAV_SCENE_BASE_UAV_TRAJECTORY_VISUALIZE": self.UAV_SCENE_BASE_UAVTrajectoryVisualize,
            "UAV_SCENE_BASE_UAVDisVisualize": self.UAV_SCENE_BASE_UAVDisVisualize,
            "UAV_SCENE_BASE_UAVAlertDisVisualize": self.UAV_SCENE_BASE_UAVAlertDisVisualize,
            "UAV_SCENE_BASE_UAVAvgDisStore": self.UAV_SCENE_BASE_UAVAvgDisStore,
            "UAV_SCENE_BASE_UAVMinDisStore": self.UAV_SCENE_BASE_UAVMinDisStore,
            "UAV_MULTI_TARGET_SCENE_BASE_UAVAlertDisStore": self.UAV_SCENE_BASE_UAVAlertDisStore,
            "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore": self.UAV_SCENE_BASE_UAVFitnessStore
        }

        if statOutputDict is not None:
            self.__UAV_Scene_Base_stat_output_dict.update(statOutputDict)

        if statOutputRegisters is None:
            statOutputRegisters = ["UAV_SCENE_BASE_UAV_TRAJECTORY_VISUALIZE"]
        self.statOutputFuncReg = statFuncListGenerator(statOutputRegisters, self.__UAV_Scene_Base_stat_output_dict)
        self.UAV_SCENE_BASE_Args = ArgsDictValueController(sceneArgs,
                                                           self.UAV_SCENE_BASE_DEFAULT_ARGS,
                                                           onlyUseDefaultKey=True)

        # self.initArgs = {"agent":{}, "target":{}, "MAS":{}}
        self._initAgents(agentsCls, agentsArgs, optimizerCls, optimizerArgs, deltaTime)
        self._initTargets(targetCls, targetArgs, deltaTime)

        self.initArgs["MAS"]["statOutputRegisters"] = statOutputRegisters
        self._initMAS(MAS_Cls, self.agents, MAS_Args, deltaTime)
        del self.initArgs["MAS"]["statOutputRegisters"]

        super().__init__(self.agents, self.multiAgentSystem, needRunningTime, sceneArgs=sceneArgs)

    def _initAgents(self, agentsCls, agentsArgs, optimizerCls, optimizerArgs, deltaTime):
        if isinstance(agentsArgs["initArgs"], list) is False:
            self.agents = [agentsCls(initPositionState=agentsArgs["initArgs"]["initPositionState"],
                                     linearVelocityRange=agentsArgs["initArgs"]["linearVelocityRange"],
                                     angularVelocityRange=agentsArgs["initArgs"]["angularVelocityRange"],
                                     agentArgs=agentsArgs["computationArgs"],
                                     optimizerCls=optimizerCls,
                                     optimizerInitArgs=optimizerArgs["optimizerInitArgs"],
                                     optimizerComputationArgs=optimizerArgs["optimizerComputationArgs"],
                                     deltaTime=deltaTime) for i in range(self.agentsNum)]
        else:
            self.agents = [agentsCls(initPositionState=agentsArgs["initArgs"][i]["initPositionState"],
                                     linearVelocityRange=agentsArgs["initArgs"][i]["linearVelocityRange"],
                                     angularVelocityRange=agentsArgs["initArgs"][i]["angularVelocityRange"],
                                     agentArgs=agentsArgs["computationArgs"],
                                     optimizerCls=optimizerCls,
                                     optimizerInitArgs=optimizerArgs["optimizerInitArgs"],
                                     optimizerComputationArgs=optimizerArgs["optimizerComputationArgs"],
                                     deltaTime=deltaTime) for i in range(self.agentsNum)]

    def _initTargets(self, targetCls, targetArgs, deltaTime):
        if self.targetNum == 1:
            self.targets = [targetCls(initPositionState=targetArgs["initPositionState"],
                                      linearVelocityRange=targetArgs["linearVelocityRange"],
                                      angularVelocityRange=targetArgs["angularVelocityRange"],
                                      movingFuncRegister=targetArgs["movingFuncRegister"],
                                      deltaTime=deltaTime)]
            self.target = self.targets[0]
        else:
            if isinstance(targetArgs, list) is False:
                self.targets = [targetCls(initPositionState=targetArgs["initPositionState"],
                                          linearVelocityRange=targetArgs["linearVelocityRange"],
                                          angularVelocityRange=targetArgs["angularVelocityRange"],
                                          movingFuncRegister=targetArgs["movingFuncRegister"],
                                          deltaTime=deltaTime) for i in range(self.targetNum)]
            else:
                self.targets = [targetCls(initPositionState=targetArgs[i]["initPositionState"],
                                          linearVelocityRange=targetArgs[i]["linearVelocityRange"],
                                          angularVelocityRange=targetArgs[i]["angularVelocityRange"],
                                          movingFuncRegister=targetArgs[i]["movingFuncRegister"],
                                          deltaTime=deltaTime) for i in range(self.targetNum)]

    def _getMASStatRegister(self, sceneRegisters=None, masRegisters=None):
        if masRegisters is not None:
            statRegisters = list(masRegisters)
        else:
            statRegisters = list()

        if sceneRegisters is not None:
            for item in sceneRegisters:
                statRegisters.append(self.SCENE_AND_MAS_RECORD_MAP[item])

        if len(sceneRegisters) == 0:
            return None
        else:
            return list(set(statRegisters))

    def _initMAS(self, MAS_Cls, agents, MAS_Args, deltaTime):

        self.multiAgentSystem = MAS_Cls(agents=agents,
                                        MAS_Args=MAS_Args,
                                        statRegisters=self._getMASStatRegister(
                                            sceneRegisters=self.initArgs["MAS"]["statOutputRegisters"]))

    def runningFinal(self):
        self.__csvDataLists = []
        self.__csvNameLists = []
        self.__statOutputFuncIsDone = False
        for item in self.statOutputFuncReg:
            # time.sleep(0.5)
            item()
        self.__statOutputFuncIsDone = True
        self.UAV_SCENE_BASE_SimpleStoreStatData(None, None)

    def runningInner(self):
        if self.targetNum == 1:
            self.multiAgentSystem.recvFromEnv(targetPosition=self.target.positionState)
        else:
            self.multiAgentSystem.recvFromEnv(targetPosition=[item.positionState for item in self.targets])

        # self.multiAgentSystem.optimization()
        self.multiAgentSystem.update()

        if self.targetNum == 1:
            self.target.update()
        else:
            for item in self.targets:
                item.update()

    '''
    following is stat data function output or visualize function
    '''

    def UAV_SCENE_BASE_SimpleVisualizeTrajectory(self, scattersList, nameList, labelNames=None, titleName=None, showOriginPoint=False,
                                                 saveFigName=None, ifScatterPlotPoint=False, figureArgs=None):
        cd = CoorDiagram()
        if self.figureSavePath is None:
            cd.drawManyScattersInOnePlane(scattersList, nameList=nameList, titleName=titleName,labelNames=labelNames,
                                          showOriginPoint=showOriginPoint, saveFigName=saveFigName,
                                          saveFigNameSuffix=self.UAV_SCENE_BASE_Args["saveFigNameSuffix"],
                                          ifScatterPlotPoint=ifScatterPlotPoint, figureArgs=figureArgs)
        else:
            cd.setStorePath(self.figureSavePath)
            cd.drawManyScattersInOnePlane(scattersList, nameList=nameList, titleName=titleName,
                                          labelNames=labelNames, ifSaveFig=True,
                                          showOriginPoint=showOriginPoint, saveFigName=saveFigName,
                                          saveFigNameSuffix=self.UAV_SCENE_BASE_Args["saveFigNameSuffix"],
                                          ifScatterPlotPoint=ifScatterPlotPoint, figureArgs=figureArgs)

    def UAV_SCENE_BASE_SimpleStoreStatData(self, scattersList, nameList):
        if self.__statOutputFuncIsDone is True:
            if self.__csvNameLists != [] and self.__csvDataLists != []:
                saveStatDataPath = "%s_%s" % (
                    self.UAV_SCENE_BASE_Args["storeStatDataName"], time.strftime("%Y%m%d_%H%M%S.csv", time.localtime()))
                if self.csvSavePath[-1] != "/":
                    self.csvSavePath = self.csvSavePath + "/"

                if self.csvSavePath is not None:
                    if os.path.exists(self.csvSavePath) is False:
                        os.mkdir(self.csvSavePath)

                    saveCSVDict = dict()
                    maxLenOfDataList = max([len(item) for item in self.__csvDataLists])
                    for j in range(maxLenOfDataList + 1):
                        for item in self.__csvDataLists:
                            if j > len(item):
                                item.append("")

                    for nameItem, dataItem in zip(self.__csvNameLists, self.__csvDataLists):
                        saveCSVDict.update({nameItem: dataItem})

                    df = pd.DataFrame(saveCSVDict)
                    df.to_csv("%s%s" % (self.csvSavePath, saveStatDataPath))
        else:
            self.__csvNameLists.extend(nameList)
            self.__csvDataLists.append(scattersList)

    def UAV_SCENE_BASE_UAVTrajectoryVisualize(self):
        scattersList = []
        nameList = []
        if self.targetNum == 1:
            scattersList.append(self.target.coordinateVector)
            nameList.append("target")
        else:
            for i, item in enumerate(self.targets):
                scattersList.append(item.coordinateVector)
                nameList.append(r"target %d" % i)

        for i, item in enumerate(self.agents):
            scattersList.append(item.coordinateVector)
            nameList.append(r"uav %d" % i)

        self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList=scattersList, nameList=nameList, titleName="uav trajectory",
                                                      showOriginPoint=True, saveFigName="Trajectory",
                                                      ifScatterPlotPoint=True)

    def UAV_SCENE_BASE_UAVDisVisualize(self):
        scattersList = []
        nameList = []

        try:
            if hasattr(self.multiAgentSystem, "UAVDisVisualizeStat"):
                for item in self.multiAgentSystem.UAVDisVisualizeStat.items():
                    scattersList.append(item[1])
                    nameList.append(item[0])
                # for i, item  in enumerate(self.agents):
                #     scattersList.append(item.coordinateVector)
                #     nameList.append(r"uav %d" % i)

                self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList=scattersList, nameList=nameList, titleName="uav distance between",
                                                              saveFigName="uavDisBtw",
                                                              figureArgs={"figSize": self.NARROW_FIGURE_SIZE})
            else:
                raise NotImplementedError("There is no variable named UAVDisVisualizeStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def UAV_SCENE_BASE_UAVAlertDisVisualize(self):
        scattersList, nameList, alertPercentage, thersholdStr = self.__UAV_SCENE_BASE_UAVAlertDisCalcInner()
        myLogger.myLogger_Logger().info("%f Gens have over the %s threshold" % (
            alertPercentage, thersholdStr))
        self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList=scattersList, nameList=nameList,
                                                      titleName="uav alert distance record", saveFigName="alertDis")

    def UAV_SCENE_BASE_UAVAlertDisStore(self):
        nameList = ["alert percentage"]
        _, _, alertPercentage, _ = self.__UAV_SCENE_BASE_UAVAlertDisCalcInner()

        self.UAV_SCENE_BASE_SimpleStoreStatData([alertPercentage], nameList)

    def UAV_SCENE_BASE_UAVAvgDisStore(self):
        scattersList = []
        nameList = ["UAV Avg Dis"]
        if hasattr(self.multiAgentSystem, "UAVAvgDisListStat"):
            for item in self.multiAgentSystem.UAVAvgDisListStat:
                scattersList.append(item[1])

            self.UAV_SCENE_BASE_SimpleStoreStatData(scattersList, nameList)
        else:
            raise NotImplementedError("There is no variable named UAVAvgDisListStat needed"
                                      "when call function %s" % (inspect.stack()[0][3]))

    def UAV_SCENE_BASE_UAVMinDisStore(self):
        scattersList = []
        nameList = ["UAV Min Dis"]
        if hasattr(self.multiAgentSystem, "UAVMinDisListStat"):
            for item in self.multiAgentSystem.UAVMinDisListStat:
                scattersList.append(item[1])

            self.UAV_SCENE_BASE_SimpleStoreStatData(scattersList, nameList)
        else:
            raise NotImplementedError("There is no variable named UAVMinDisListStat needed"
                                      "when call function %s" % (inspect.stack()[0][3]))

    def UAV_SCENE_BASE_UAVFitnessStore(self):
        scattersList = []
        nameList = ["fitness", "stability of fitness"]
        try:
            if hasattr(self.multiAgentSystem, "fitnessStat"):
                scattersList.append([item[1] for item in self.multiAgentSystem.fitnessStat])
                scattersList.append([np.average(
                    [abs(1 / self.multiAgentSystem.fitnessStat[index][1] - 1 /
                         self.multiAgentSystem.fitnessStat[index - 1][1]) for
                     index in range(1, len(self.multiAgentSystem.fitnessStat))])])

                for i in range(len(nameList)):
                    self.UAV_SCENE_BASE_SimpleStoreStatData(scattersList[i], [nameList[i]])
            else:
                raise NotImplementedError("There is no variable named fitnessStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def __UAV_SCENE_BASE_UAVAlertDisCalcInner(self):
        scattersList = []
        nameList = []

        try:
            if hasattr(self.multiAgentSystem, "UAVAlertDisVisualizeStat"):
                alertPercentage = 0.
                thersholdStr = ""
                for item in self.multiAgentSystem.UAVAlertDisVisualizeStat.items():
                    scattersList.append(item[1])
                    nameList.append(item[0])
                    itemUAVAlertDisArray = np.vstack(item[1])[:, 1]
                    alertPercentage = np.sum(itemUAVAlertDisArray) / itemUAVAlertDisArray.size
                    thersholdStr = item[0]

                return scattersList, nameList, alertPercentage, thersholdStr

            else:
                raise NotImplementedError("There is no variable named UAVDisVisualizeStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))
