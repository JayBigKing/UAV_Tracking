#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_Scene_Base.py
@Author  : jay.zhu
@Time    : 2022/11/5 14:06
"""
import inspect

import numpy as np

from Jay_Tool.LogTool import myLogger
from Scene.UAV_Scene.UAV_Scene_Base import UAV_Scene_Base


class UAV_MultiTarget_Scene_Base(UAV_Scene_Base):
    DIS_BETWEEN_TARGET_AGENT_RANGE_LIST = [[0., 10.], [10., 20.], [20., 30.], [30., 40.]]
    DEFAULT_MAS_RECORD_REG = ["recordDisOfUAVsForVisualize", "recordNumOfTrackingUAVForTarget",
                              "recordAlertDisOfUAVsForVisualize", "recordDisBetweenTargetAndUAV"]

    def __init__(self, agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                 MAS_Args, needRunningTime, targetNum=1, deltaTime=1., figureSavePath=None,
                 userStatOutputRegisters=None, sceneArgs=None):
        self.__UAV_MULTI_TARGET_SCENE_BASE_STAT_OUTPUT_DICT = {
            "UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VISUALIZE": self.UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VISUALIZE,
            "UAV_MULTI_TARGET_SCENE_BASE_UAV_CONSUME_VISUALIZE": self.UAV_MULTITARGET_SCENE_BASE_UAV_CONSUME_VISUALIZE,
            "UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_VISUALIZE": self.UAV_MULTITARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_VISUALIZE,
            "UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE": self.UAV_MULTITARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE,
            "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STORE": self.UAV_MULTITARGET_SCENE_BASE_AVG_DIS_STORE,
            "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE": self.UAV_MULTITARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE,
            "UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_STORE": self.UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_STORE,
            "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_STORE": self.UAV_MULTITARGET_SCENE_BASE_EFFECTIVE_TIME_STORE,
            "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_FOR_TARGET_STORE": self.UAV_MULTITARGET_SCENE_BASE_EFFECTIVE_TIME_FOR_TARGET_STORE,
            "UAV_MULTI_TARGET_SCENE_BASE_TRACK_TARGET_ID_STORE": self.UAV_MULTITARGET_SCENE_BASE_TRACK_TARGET_ID_STORE,
            "UAV_MULTI_TARGET_SCENE_BASE_AVG_CLOSE_DIS_STORE": self.UAV_MULTITARGET_SCENE_BASE_AVG_CLOSE_DIS_STORE,
            "UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_VISUALIZE": self.UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_VISUALIZE

        }
        self.SCENE_AND_MAS_RECORD_MAP.update({
            "UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE": "recordDisBetweenTargetAndUAV",
            "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STORE": "recordDisBetweenTargetAndUAV",
            "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE": "recordDisBetweenTargetAndUAV",
            "UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_STORE": "recordNumOfTrackingUAVForTarget",
            "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_STORE": "recordEffectiveTime",
            "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_FOR_TARGET_STORE": "recordEffectiveTimeForTarget",
            "UAV_MULTI_TARGET_SCENE_BASE_UAVAlertDisStore": "recordAlertDisOfUAVsForVisualize",
            "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore": "recordFitness",
            "UAV_MULTI_TARGET_SCENE_BASE_TRACK_TARGET_ID_STORE": "recordTrackTargetID",
            "UAV_MULTI_TARGET_SCENE_BASE_AVG_CLOSE_DIS_STORE": "recordDisBetweenCloseTar_UAV",
            "UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_VISUALIZE": "recordNumOfTrackingUAVForTarget"
        })

        statOutputRegisters = ["UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_VISUALIZE",
                               "UAV_SCENE_BASE_UAVDisVisualize",
                               "UAV_SCENE_BASE_UAV_TRAJECTORY_VISUALIZE",
                               "UAV_SCENE_BASE_UAVAlertDisVisualize",
                               "UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_VISUALIZE", ]
        if userStatOutputRegisters is not None:
            statOutputRegisters.extend(userStatOutputRegisters)

        self.initArgs["MAS"]["userStatOutputRegisters"] = userStatOutputRegisters
        super().__init__(agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                         MAS_Args, needRunningTime, targetNum, deltaTime, figureSavePath, statOutputRegisters,
                         self.__UAV_MULTI_TARGET_SCENE_BASE_STAT_OUTPUT_DICT, sceneArgs=sceneArgs)
        del self.initArgs["MAS"]["userStatOutputRegisters"]

    def _initAgents(self, agentsCls, agentsArgs, optimizerCls, optimizerArgs, deltaTime):
        if isinstance(agentsArgs["initArgs"], list) is False:
            self.agents = [agentsCls(initPositionState=agentsArgs["initArgs"]["initPositionState"],
                                     linearVelocityRange=agentsArgs["initArgs"]["linearVelocityRange"],
                                     angularVelocityRange=agentsArgs["initArgs"]["angularVelocityRange"],
                                     agentArgs=agentsArgs["computationArgs"],
                                     optimizerCls=optimizerCls,
                                     optimizerInitArgs=optimizerArgs["optimizerInitArgs"],
                                     optimizerComputationArgs=optimizerArgs["optimizerComputationArgs"],
                                     targetNum=self.targetNum,
                                     deltaTime=deltaTime) for i in range(self.agentsNum)]
        else:
            self.agents = [agentsCls(initPositionState=agentsArgs["initArgs"][i]["initPositionState"],
                                     linearVelocityRange=agentsArgs["initArgs"][i]["linearVelocityRange"],
                                     angularVelocityRange=agentsArgs["initArgs"][i]["angularVelocityRange"],
                                     agentArgs=agentsArgs["computationArgs"],
                                     optimizerCls=optimizerCls,
                                     optimizerInitArgs=optimizerArgs["optimizerInitArgs"],
                                     optimizerComputationArgs=optimizerArgs["optimizerComputationArgs"],
                                     targetNum=self.targetNum,
                                     deltaTime=deltaTime) for i in range(self.agentsNum)]

    def _initTargets(self, targetCls, targetArgs, deltaTime):
        if isinstance(targetArgs, list) is False:
            self.targets = [targetCls(initPositionState=targetArgs["initPositionState"],
                                      linearVelocityRange=targetArgs["linearVelocityRange"],
                                      angularVelocity=targetArgs["angularVelocityRange"],
                                      movingFuncRegister=targetArgs["movingFuncRegister"],
                                      deltaTime=deltaTime) for i in range(self.targetNum)]
        else:
            self.targets = [targetCls(initPositionState=targetArgs[i]["initPositionState"],
                                      linearVelocityRange=targetArgs[i]["linearVelocityRange"],
                                      angularVelocityRange=targetArgs[i]["angularVelocityRange"],
                                      movingFuncRegister=targetArgs[i]["movingFuncRegister"],
                                      deltaTime=deltaTime) for i in range(self.targetNum)]

    def _initMAS(self, MAS_Cls, agents, MAS_Args, deltaTime):
        self.multiAgentSystem = MAS_Cls(agents=agents,
                                        masArgs=MAS_Args,
                                        targetNum=self.targetNum,
                                        statRegisters=self._getMASStatRegister(
                                            sceneRegisters=self.initArgs["MAS"]["userStatOutputRegisters"]),
                                        deltaTime=deltaTime)

    def runningInner(self):
        self.multiAgentSystem.recvFromEnv(targetPosition=[item.positionState for item in self.targets])

        self.multiAgentSystem.update()

        for item in self.targets:
            item.update()

    '''
    following is stat data function output or visualize function
    '''

    def UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VISUALIZE(self):
        scattersList = []
        nameList = []
        try:
            if hasattr(self.multiAgentSystem, "numOfTrackingUAVForTargetStat"):
                numOfTrackingUAVForTargetStat = self.multiAgentSystem.numOfTrackingUAVForTargetStat
                for i, item in enumerate(numOfTrackingUAVForTargetStat):
                    scattersList.append(item)
                    nameList.append(r"target %d" % i)

                self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList=scattersList, nameList=nameList, titleName="num of each target"
                                                                                                "tracked by uav",
                                                              showOriginPoint=False, saveFigName="TrackTarNum")
            else:
                raise NotImplementedError("There is no variable named consumeOfEachUAVStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def __UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_INNER(self):
        varianceInEpochList = []
        try:
            if hasattr(self.multiAgentSystem, "numOfTrackingUAVForTargetStat"):
                numOfTrackingUAVForTargetStat = self.multiAgentSystem.numOfTrackingUAVForTargetStat

                for j in range(len(numOfTrackingUAVForTargetStat[0])):
                    varianceInEpochList.append(np.var([float(item[j][1]) for item in numOfTrackingUAVForTargetStat]))

                return varianceInEpochList
            else:
                raise NotImplementedError("There is no variable named numOfTrackingUAVForTargetStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_VISUALIZE(self):
        varianceInEpochList = self.__UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_INNER()
        scattersList = [[np.array([float(index), item]) for index, item in enumerate(varianceInEpochList)]]
        self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList=scattersList, nameList=None, showOriginPoint=False,
                                                      saveFigName="Variance",
                                                      figureArgs={"figSize": self.NARROW_FIGURE_SIZE})

    def UAV_MULTITARGET_SCENE_BASE_UAV_CONSUME_VISUALIZE(self):
        scattersList = []
        nameList = []
        try:
            if hasattr(self.multiAgentSystem, "consumeOfEachUAVStat"):
                consumeOfEachUAVStat = self.multiAgentSystem.consumeOfEachUAVStat
                for i, item in enumerate(consumeOfEachUAVStat):
                    scattersList.append(item)
                    nameList.append(r"uav %d" % i)

                self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList=scattersList, nameList=nameList, titleName="each uav consume",
                                                              showOriginPoint=False, saveFigName="consume")
            else:
                raise NotImplementedError("There is no variable named consumeOfEachUAVStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def UAV_MULTITARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_VISUALIZE(self):
        scattersList = []
        nameList = []
        try:
            if hasattr(self.multiAgentSystem, "disBetweenTargetAndUAVStat"):
                disBetweenTargetAndUAVStat = self.multiAgentSystem.disBetweenTargetAndUAVStat
                for i, item in enumerate(disBetweenTargetAndUAVStat):
                    scattersList.append(item)
                    nameList.append(r"distance between uav %d and target" % i)

                self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList=scattersList, nameList=nameList,
                                                              titleName="distance between uav and targets",
                                                              showOriginPoint=False,
                                                              saveFigName="disBtwTarAndUAV")
            else:
                raise NotImplementedError("There is no variable named disBetweenTargetAndUAV needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def UAV_MULTITARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE(self):
        scattersList = [[]]
        nameList = ["distance between uav and target", "dis Between Target-Agent Range", "samplePoints"]
        try:
            if hasattr(self.multiAgentSystem, "disBetweenTargetAndUAVStat"):
                disBetweenTargetAndUAVStat = self.multiAgentSystem.disBetweenTargetAndUAVStat
                for item in disBetweenTargetAndUAVStat:
                    scattersList[0].extend([itemInner[1] for itemInner in item])

                scattersList.append(
                    [r'[%d, %d]' % (int(item[0]), int(item[1])) for item in self.DIS_BETWEEN_TARGET_AGENT_RANGE_LIST])
                scattersList[1].append('others')
                scattersList.append([0 for j in range(len(self.DIS_BETWEEN_TARGET_AGENT_RANGE_LIST) + 1)])
                for item in scattersList[0]:
                    otherFlag = True
                    for rangeIndex, rangeItem in enumerate(self.DIS_BETWEEN_TARGET_AGENT_RANGE_LIST):
                        if rangeItem[0] <= item < rangeItem[1]:
                            scattersList[-1][rangeIndex] += 1
                            otherFlag = False
                            break

                    if otherFlag is True:
                        scattersList[-1][-1] += 1

                for i in range(len(nameList)):
                    self.UAV_SCENE_BASE_SimpleStoreStatData(scattersList[i], [nameList[i]])
            else:
                raise NotImplementedError("There is no variable named disBetweenTargetAndUAV needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def __UAV_MULTITARGET_SCENE_BASE_AVG_DIS_CALC_INNER(self):
        scattersList = []
        try:
            if hasattr(self.multiAgentSystem, "disBetweenTargetAndUAVStat"):
                disBetweenTargetAndUAVStat = self.multiAgentSystem.disBetweenTargetAndUAVStat

                for j in range(len(disBetweenTargetAndUAVStat[0])):
                    scattersList.append(np.average([item[j][1] for item in disBetweenTargetAndUAVStat]))

                return scattersList
            else:
                raise NotImplementedError("There is no variable named disBetweenTargetAndUAV needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def UAV_MULTITARGET_SCENE_BASE_AVG_DIS_STORE(self):
        scattersList = self.__UAV_MULTITARGET_SCENE_BASE_AVG_DIS_CALC_INNER()
        nameList = ["avg distance between uav and target"]
        self.UAV_SCENE_BASE_SimpleStoreStatData(scattersList, nameList)

    def UAV_MULTITARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE(self):
        avgDisList = self.__UAV_MULTITARGET_SCENE_BASE_AVG_DIS_CALC_INNER()
        nameList = ["stability of avg distance between uav and target"]

        if avgDisList == [] or len(avgDisList) == 1:
            return

        diffBetweenEpochsList = [abs(avgDisList[i] - avgDisList[i - 1]) for i in range(1, len(avgDisList))]
        AvgOfDiffBetweenEpochs = np.average(diffBetweenEpochsList)
        stabilityStat = np.average([(item - AvgOfDiffBetweenEpochs) ** 2 for item in diffBetweenEpochsList])

        self.UAV_SCENE_BASE_SimpleStoreStatData([stabilityStat], nameList)

    def UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_STORE(self):
        varianceInEpochList = self.__UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_INNER()
        nameList = ["variance of target tracked num"]

        self.UAV_SCENE_BASE_SimpleStoreStatData(varianceInEpochList, nameList)

    def UAV_MULTITARGET_SCENE_BASE_EFFECTIVE_TIME_STORE(self):
        nameList = ["max effective time", "avg effective time", "min effective time"]
        try:
            if hasattr(self.multiAgentSystem, "effectiveTimeStat"):
                effectiveTimeStat = self.multiAgentSystem.effectiveTimeStat
                effectiveTimeList = []
                for item in effectiveTimeStat:
                    effectiveTimeList.extend(item)
                if effectiveTimeList != []:
                    scattersList = [max(effectiveTimeList), np.average(effectiveTimeList), min(effectiveTimeList)]
                else:
                    scattersList = [0., 0., 0.]

                for i in range(len(nameList)):
                    self.UAV_SCENE_BASE_SimpleStoreStatData([scattersList[i]], [nameList[i]])

            else:
                raise NotImplementedError("There is no variable named effectiveTimeStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def UAV_MULTITARGET_SCENE_BASE_EFFECTIVE_TIME_FOR_TARGET_STORE(self):
        scattersList = []
        nameList = ["effectively track ratio", "effectivePointList", "avg effective time for target"]
        try:
            if hasattr(self.multiAgentSystem, "effectiveTimeForTargetStat"):
                effectiveTimeForTargetStat = self.multiAgentSystem.effectiveTimeForTargetStat
                effectivePointForTargetStat = self.multiAgentSystem.effectivePointForTargetStat
                effectiveTimeList = []
                effectivePointList = []
                for item in effectiveTimeForTargetStat:
                    effectiveTimeList.extend(item)
                for item in effectivePointForTargetStat:
                    effectivePointList.extend(item)

                if effectivePointList != []:
                    scattersList.append([float(sum(effectivePointList)) / float(len(effectivePointList))])
                    scattersList.append([0.])
                else:
                    scattersList.append([0.])
                    scattersList.append([])

                if effectiveTimeList != []:
                    scattersList.append([np.average(effectiveTimeList)])
                else:
                    scattersList.append([0.])

                for i in range(len(nameList)):
                    self.UAV_SCENE_BASE_SimpleStoreStatData(scattersList[i], [nameList[i]])

            else:
                raise NotImplementedError("There is no variable named effectiveTimeForTargetStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def UAV_MULTITARGET_SCENE_BASE_TRACK_TARGET_ID_STORE(self):
        nameListFmt = "uav {:d} track target ID"
        nameList = []
        try:
            if hasattr(self.multiAgentSystem, "trackTargetIDStat"):
                for index, item in enumerate(self.multiAgentSystem.trackTargetIDStat):
                    nameList = [nameListFmt.format(index)]
                    scattersList = [item2[1] for item2 in item]

                    self.UAV_SCENE_BASE_SimpleStoreStatData(scattersList, nameList)

            else:
                raise NotImplementedError("There is no variable named trackTargetIDStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))

    def UAV_MULTITARGET_SCENE_BASE_AVG_CLOSE_DIS_STORE(self):
        scattersList = []
        nameList = ["avg distance between uav and closest target"]
        try:
            if hasattr(self.multiAgentSystem, "disBetweenCloseTarUAVStat"):
                disBetweenCloseTarUAVStat = self.multiAgentSystem.disBetweenCloseTarUAVStat

                for j in range(len(disBetweenCloseTarUAVStat[0])):
                    scattersList.append(np.average([item[j][1] for item in disBetweenCloseTarUAVStat]))

                self.UAV_SCENE_BASE_SimpleStoreStatData(scattersList, nameList)

            else:
                raise NotImplementedError("There is no variable named disBetweenCloseTarUAVStat needed"
                                          "when call function %s" % (inspect.stack()[0][3]))
        except NotImplementedError as e:
            myLogger.myLogger_Logger().warn(repr(e))
