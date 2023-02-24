#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : experimentForPaperMP.py
@Author  : jay.zhu
@Time    : 2023/1/22 1:04
"""
import sys
import os
import time

sys.path.append("../../../")
from experiment.experimentInit import experimentInit
from Jay_Tool.LogTool import myLogger
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_Agent import UAV_MultiTarget_Agent
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_ProbabilitySelectTargetAgent import \
    UAV_MultiTargets_ProbabilitySelectTargetAgent
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_MPC import UAV_MultiTargets_MPC
import experimentBase
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe, Manager
from Scene.UAV_Scene.multiTarget.UAV_MultiTarget_UsingDatasetScene import UAV_MultiTarget_PredictScene, \
    UAV_MultiTarget_UsingDatasetScene

saveFigPathPrefix = "../../experimentRes/experimentForPaper/"

experimentBase.NEED_RUNNING_TIME = 200


class ExperimentBase(ABC):
    class UAV_MTMP_PredictScene(UAV_MultiTarget_PredictScene):
        def __init__(self, agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs,
                     MAS_Cls,
                     MAS_Args, needRunningTime, predictorCls=None, targetNum=1, deltaTime=1., figureSavePath=None,
                     userStatOutputRegisters=None, sceneArgs=None):
            super().__init__(agentsNum=agentsNum,
                             agentsCls=agentsCls,
                             agentsArgs=agentsArgs,
                             optimizerCls=optimizerCls,
                             optimizerArgs=optimizerArgs,
                             targetCls=targetCls,
                             targetArgs=targetArgs,
                             MAS_Cls=MAS_Cls,
                             MAS_Args=MAS_Args,
                             needRunningTime=needRunningTime,
                             predictorCls=predictorCls,
                             targetNum=targetNum,
                             deltaTime=deltaTime,
                             figureSavePath=figureSavePath,
                             userStatOutputRegisters=userStatOutputRegisters,
                             sceneArgs=sceneArgs)

        def runMultiProcess(self, managerDict, managerList, currentProgressHandler, nowSceneIndex, sceneListLength,
                            clsName):
            self.runningPreProcess()
            while self.shouldContinue():
                managerDict["currentProgress"] = currentProgressHandler(self.nowRunningTime, self.needRunningTimes,
                                                                        nowSceneIndex, sceneListLength, clsName)
                self.runningInner()
            self.runningFinal()

    class UAV_MTMP_UsingDatasetScene(UAV_MultiTarget_UsingDatasetScene):
        def __init__(self, UAV_Dataset, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, MAS_Cls,
                     MAS_Args, needRunningTime, predictorCls=None, deltaTime=1., figureSavePath=None,
                     userStatOutputRegisters=None,
                     sceneArgs=None):
            super().__init__(UAV_Dataset=UAV_Dataset,
                             agentsCls=agentsCls,
                             agentsArgs=agentsArgs,
                             optimizerCls=optimizerCls,
                             optimizerArgs=optimizerArgs,
                             targetCls=targetCls,
                             MAS_Cls=MAS_Cls,
                             MAS_Args=MAS_Args,
                             needRunningTime=needRunningTime,
                             predictorCls=predictorCls,
                             deltaTime=deltaTime,
                             figureSavePath=figureSavePath,
                             userStatOutputRegisters=userStatOutputRegisters,
                             sceneArgs=sceneArgs)

        def runMultiProcess(self, managerDict, managerList, currentProgressHandler, nowSceneIndex, sceneListLength,
                            clsName):
            self.runningPreProcess()
            while self.shouldContinue():
                managerDict["currentProgress"] = currentProgressHandler(self.nowRunningTime, self.needRunningTimes,
                                                                        nowSceneIndex, sceneListLength, clsName)
                self.runningInner()
            self.runningFinal()

    def __init__(self):
        self.sceneList = []
        self.newDatasetPathList = []
        self.nowSceneIndex = 0
        self.terminalFlag = False
        self.nowClsName = self.getClsName()
        if self.nowClsName is None:
            raise NotImplementedError("Please implement the method named getClsName")
        self.getUAVSceneList()

    @abstractmethod
    def getClsName(self):
        return None

    def getUAVSceneList(self):
        self.getUAVSceneListInner()
        # self.packSceneClsForRunningTime()

    @abstractmethod
    def getUAVSceneListInner(self):
        pass

    # def packSceneClsForRunningTime(self):
    #     self.nowRunningTimeList = []
    #     if self.sceneList != []:
    #         for index, item in enumerate(self.sceneList):
    #             nowRunningTimePtr = [0.]
    #             self.nowRunningTimeList.append(nowRunningTimePtr)
    #             item.nowRunningTimeList = nowRunningTimePtr
    #             # item.shouldContinue = shouldContinueForMP
    #             # item.__getattribute__ = self.get_attr
    #             # item.__setattr__ = self.set_attr

    @staticmethod
    def currentProgressHandler(nowRunningTime, needRunningTime, nowSceneIndex, sceneListLength, clsName):
        currentProgress = (needRunningTime * nowSceneIndex +
                           nowRunningTime) / (needRunningTime * sceneListLength)
        return "%s currentProgress: %f" % (clsName, currentProgress)

    def process(self, managerDict, managerList):
        if self.nowClsName is not None:
            workProcessHandler = Process(target=self.workProcess,
                                         name='%s_%s' % (self.nowClsName, self.workProcess.__name__),
                                         args=(managerDict, managerList))
            workProcessHandler.start()

    # def progressTrackedProcess(self, managerDict, managerList):
    #     while True:
    #         time.sleep(1)
    #         currentProgress = (experimentBase.NEED_RUNNING_TIME * self.nowSceneIndex +
    #                            self.sceneList[self.nowSceneIndex].nowRunningTime) / (
    #                                   experimentBase.NEED_RUNNING_TIME * len(self.sceneList))
    #         managerDict["currentProgress"] = "%s currentProgress: %f" % (self.nowClsName, currentProgress)
    #         managerDict["terminalFlag"] = self.terminalFlag
    @clockTester
    def workProcess(self, managerDict, managerList):
        experimentInit()
        self.workProcessInner(managerDict, managerList)
        self.workProcessFinal(managerDict, managerList)

    def workProcessInner(self, managerDict, managerList):
        for item in self.sceneList:
            managerDict["terminalFlag"] = False
            item.runMultiProcess(managerDict, managerList, self.currentProgressHandler, self.nowSceneIndex,
                                 len(self.sceneList), self.nowClsName)
            self.nowSceneIndex += 1

    def workProcessFinal(self, managerDict, managerList):
        if self.newDatasetPathList is not None:
            for item in set(self.newDatasetPathList):
                os.remove(item)

        self.terminalFlag = True
        managerDict["currentProgress"] = "%s currentProgress: %f" % (self.nowClsName, 1.)
        managerDict["terminalFlag"] = self.terminalFlag



class Experiment1(ExperimentBase):
    '''
    定性分析，获取跟踪图，和target的平均距离，UAV之间的平均距离，跟踪target的UAV数量
    '''

    def getClsName(self):
        return self.__class__.__name__

    def getUAVSceneListInner(self):
        movingWayList = [["straight", "movingStraightly"], ["sin", "movingAsSin"], ["rand", "randMoving"]]
        for item in movingWayList:
            figureSavePathKeyword = "%s%s_%s" % (saveFigPathPrefix, self.nowClsName, item[0])
            # myLogger.myLogger_Logger().info("experiment1 : %s start" % item[0])
            uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                                      getTargetTrajectoryWay="online",
                                                      masKey="PredictAndNashMAS",
                                                      optimizerKey="dynEC",
                                                      agentsNum=4,
                                                      targetNum=2,
                                                      targetMovingWay=item[1],
                                                      figureSavePathKeyword=figureSavePathKeyword,
                                                      userStatOutputRegisters=[],
                                                      uavSceneCls=ExperimentBase.UAV_MTMP_PredictScene,
                                                      sceneArgs={"ifPrintRunningEpoch": False,
                                                                 "storeStatDataName":"%s"%(item[0])})
            self.sceneList.append(uav_scene)


class Experiment2(ExperimentBase):
    '''
    定量分析，是否平衡追踪一台target的uav数量，有效跟踪占比，平均有效跟踪时间
    '''

    def getClsName(self):
        return self.__class__.__name__

    def experiment2Inner(self, agentsNum=4, targetNum=2, targetMovingWay="randMoving", runningTimes=1):
        for i in range(runningTimes):
            newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
            figureSavePathKeyword = "%s%s_a%d_t%d" % (saveFigPathPrefix, "%s_balance" % (self.nowClsName), agentsNum, targetNum)
            uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                                      getTargetTrajectoryWay="dataset",
                                                      masKey="PredictAndNashMAS",
                                                      optimizerKey="dynEC",
                                                      datasetPath=newDatasetPath,
                                                      targetMovingWay="randMoving",
                                                      figureSavePathKeyword=figureSavePathKeyword,
                                                      sceneArgs={"ifPrintRunningEpoch": False,
                                                                 "storeStatDataName":"balance"},
                                                      uavSceneCls=ExperimentBase.UAV_MTMP_UsingDatasetScene,
                                                      userStatOutputRegisters=[
                                                          "UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE",
                                                          "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_FOR_TARGET_STORE"])
            self.sceneList.append(uav_scene)

            figureSavePathKeyword = "%s%s_a%d_t%d" % (saveFigPathPrefix, "%s_no_balance" % (self.nowClsName), agentsNum, targetNum)
            agentComputationArgs = dict(experimentBase.S_DEFAULT_AGENT_COMPUTATION_ARGS)
            agentComputationArgs["JBalanceFactor"] = 0.

            uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                                      getTargetTrajectoryWay="dataset",
                                                      masKey="PredictAndNashMAS",
                                                      optimizerKey="dynEC",
                                                      datasetPath=newDatasetPath,
                                                      targetMovingWay="randMoving",
                                                      agentComputationArgs=agentComputationArgs,
                                                      figureSavePathKeyword=figureSavePathKeyword,
                                                      sceneArgs={"ifPrintRunningEpoch": False,
                                                                 "storeStatDataName":"noBalance"},
                                                      uavSceneCls=ExperimentBase.UAV_MTMP_UsingDatasetScene,
                                                      userStatOutputRegisters=[
                                                          "UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE",
                                                          "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_FOR_TARGET_STORE"])
            self.sceneList.append(uav_scene)
            self.newDatasetPathList.append(newDatasetPath)

    def getUAVSceneListInner(self):
        experiment2RunningTimes = 1
        randMovingNumList = [[4, 5], [5, 5], [6, 5]]
        moveAsSinNumList = randMovingNumList
        for index, item in enumerate(randMovingNumList):
            self.experiment2Inner(agentsNum=item[0],
                                  targetNum=item[1],
                                  targetMovingWay="randMoving",
                                  runningTimes=experiment2RunningTimes)

        for item in moveAsSinNumList:
            self.experiment2Inner(agentsNum=item[0],
                                  targetNum=item[1],
                                  targetMovingWay="movingAsSin",
                                  runningTimes=experiment2RunningTimes)

class Experiment3(ExperimentBase):
    '''
    定量分析，是否使用纳什优化的uav之间距离，alert等
    '''
    def getClsName(self):
        return self.__class__.__name__

    def experiment3Inner(self, agentsNum=4, targetNum=2, targetMovingWay="randMoving", runningTimes=1):
        useNashList = [["nash", "PredictAndNashMAS"], ["noNash", "PredictMAS"]]
        for i in range(runningTimes):
            newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
            for item in useNashList:
                figureSavePathKeyword = "%s%s_%s" % (saveFigPathPrefix, self.nowClsName, item[0])
                uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                                          getTargetTrajectoryWay="dataset",
                                                          masKey=item[1],
                                                          optimizerKey="dynEC",
                                                          datasetPath=newDatasetPath,
                                                          figureSavePathKeyword=figureSavePathKeyword,
                                                          sceneArgs={"ifPrintRunningEpoch": False,
                                                                     "storeStatDataName":"%s"%(item[0])},
                                                          uavSceneCls=ExperimentBase.UAV_MTMP_UsingDatasetScene,
                                                          userStatOutputRegisters=[
                                                              "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STORE",
                                                              "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE",
                                                              "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore",
                                                              "UAV_MULTI_TARGET_SCENE_BASE_UAVAlertDisStore"
                                                          ])
                self.sceneList.append(uav_scene)
                self.newDatasetPathList.append(newDatasetPath)

    def getUAVSceneListInner(self):
        runningTimes = 2
        self.experiment3Inner(agentsNum=4,
                              targetNum=2,
                              targetMovingWay="randMoving",
                              runningTimes=runningTimes,
                              )
        self.experiment3Inner(agentsNum=4,
                              targetNum=2,
                              targetMovingWay="movingAsSin",
                              runningTimes=runningTimes,
                              )


class Experiment4(ExperimentBase):
    '''
    定量分析，是否使用DEO的与uav的平均距离，fitness，稳定性
    '''
    __USING_AGENT_DICT = {
        "dynEC":UAV_MultiTargets_MPC,
        "noDynEC": UAV_MultiTargets_ProbabilitySelectTargetAgent,
        "ADE": UAV_MultiTargets_ProbabilitySelectTargetAgent,
        "DE": UAV_MultiTargets_ProbabilitySelectTargetAgent,
        "PSO": UAV_MultiTargets_ProbabilitySelectTargetAgent,
        "DynDE": UAV_MultiTargets_MPC
    }
    def getClsName(self):
        return self.__class__.__name__

    def experiment4Inner(self, agentsNum=4, targetNum=2, targetMovingWay="randMoving", runningTimes=1):
        experimentOptimizerList = ["dynEC", "noDynEC", "ADE", "DE", "PSO", "DynDE"]

        for i in range(runningTimes):
            newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
            for item in experimentOptimizerList:
                figureSavePathKeyword = "%s%s_%s" % (saveFigPathPrefix, self.nowClsName, item)
                uav_scene = experimentBase.experimentBase(agentCls=self.__USING_AGENT_DICT[item],
                                                          getTargetTrajectoryWay="dataset",
                                                          masKey="PredictAndNashMAS",
                                                          optimizerKey=item,
                                                          datasetPath=newDatasetPath,
                                                          figureSavePathKeyword=figureSavePathKeyword,
                                                          sceneArgs={"ifPrintRunningEpoch": False,
                                                                     "storeStatDataName":"%s"%(item)},
                                                          uavSceneCls=ExperimentBase.UAV_MTMP_UsingDatasetScene,
                                                          userStatOutputRegisters=[
                                                              "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STORE",
                                                              "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE",
                                                              "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore"
                                                          ])
                self.sceneList.append(uav_scene)
                self.newDatasetPathList.append(newDatasetPath)

    def getUAVSceneListInner(self):
        runningTimes = 2
        self.experiment4Inner(agentsNum=4,
                              targetNum=2,
                              targetMovingWay="randMoving",
                              runningTimes=runningTimes,
                              )
        self.experiment4Inner(agentsNum=4,
                              targetNum=2,
                              targetMovingWay="movingAsSin",
                              runningTimes=runningTimes,
                              )

class Experiment5(ExperimentBase):
    '''
    定量分析，是否使用概率型agent
    '''
    def getClsName(self):
        return self.__class__.__name__

    def experiment5Inner(self, agentsNum=4, targetNum=2, targetMovingWay="randMoving", runningTimes=1):
        ifUseProList = [["pro", UAV_MultiTargets_ProbabilitySelectTargetAgent], ["noPro", UAV_MultiTarget_Agent]]
        for i in range(runningTimes):
            newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
            for item in ifUseProList:
                figureSavePathKeyword = "%s%s_%s" % (saveFigPathPrefix, self.nowClsName, item[0])
                uav_scene = experimentBase.experimentBase(agentCls=item[1],
                                                          getTargetTrajectoryWay="dataset",
                                                          masKey="PredictMAS",
                                                          optimizerKey="dynEC",
                                                          datasetPath=newDatasetPath,
                                                          figureSavePathKeyword=figureSavePathKeyword,
                                                          sceneArgs={"ifPrintRunningEpoch": False,
                                                                     "storeStatDataName":"%s"%(item[0])},
                                                          uavSceneCls=ExperimentBase.UAV_MTMP_UsingDatasetScene,
                                                          userStatOutputRegisters=[
                                                              "UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_STORE"
                                                          ])
                self.sceneList.append(uav_scene)
                self.newDatasetPathList.append(newDatasetPath)

    def getUAVSceneListInner(self):
        runningTimes = 2
        self.experiment5Inner(agentsNum=4,
                              targetNum=2,
                              targetMovingWay="randMoving",
                              runningTimes=runningTimes,
                              )
        self.experiment5Inner(agentsNum=4,
                              targetNum=2,
                              targetMovingWay="movingAsSin",
                              runningTimes=runningTimes,
                              )

class Experiment6(ExperimentBase):
    '''
    定量分析，是否使用MPC
    '''
    def getClsName(self):
        return self.__class__.__name__

    def experiment5Inner(self, agentsNum=4, targetNum=2, targetMovingWay="randMoving", runningTimes=1):
        ifUseMPCList = [["MPC", UAV_MultiTargets_MPC], ["noMPC", UAV_MultiTargets_ProbabilitySelectTargetAgent]]
        for i in range(runningTimes):
            newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
            for item in ifUseMPCList:
                figureSavePathKeyword = "%s%s_%s" % (saveFigPathPrefix, self.nowClsName, item[0])
                uav_scene = experimentBase.experimentBase(agentCls=item[1],
                                                          getTargetTrajectoryWay="dataset",
                                                          masKey="PredictAndNashMAS",
                                                          optimizerKey="dynEC",
                                                          datasetPath=newDatasetPath,
                                                          figureSavePathKeyword=figureSavePathKeyword,
                                                          sceneArgs={"ifPrintRunningEpoch": False,
                                                                     "storeStatDataName":"%s"%(item[0])},
                                                          uavSceneCls=ExperimentBase.UAV_MTMP_UsingDatasetScene,
                                                          userStatOutputRegisters=[
                                                              "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STORE",
                                                              "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE",
                                                              "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore"
                                                          ])
                self.sceneList.append(uav_scene)
                self.newDatasetPathList.append(newDatasetPath)

    def getUAVSceneListInner(self):
        runningTimes = 2
        self.experiment5Inner(agentsNum=4,
                              targetNum=2,
                              targetMovingWay="randMoving",
                              runningTimes=runningTimes,
                              )
        self.experiment5Inner(agentsNum=4,
                              targetNum=2,
                              targetMovingWay="movingAsSin",
                              runningTimes=runningTimes,
                              )

@clockTester
def experimentMultiProManager():
    experimentInit()
    with Manager() as manager:
        # classList = [Experiment1, Experiment2, Experiment3, Experiment4, Experiment5, Experiment6]
        classList = [Experiment4]
        dictList = [manager.dict() for item in classList]
        objList = [item() for item in classList]
        processList = [Process(target=item.process, name=item.getClsName(), args=(dictList[index], None,)) for
                       index, item in
                       enumerate(objList)]
        terminalFlagList = [False for item in classList]
        currentProgressStr = ["%s currentProgress: 0.0" % item.getClsName() for item in objList]

        outputStr = "\r\n*****************************\r\n" \
                    "%s\r\n" \
                    "%s\r\n" \
                    "*****************************\r\n"

        for item in processList:
            item.start()

        terminalCount = 0
        while True:
            time.sleep(10)
            if terminalCount >= len(objList):
                break

            for index, item in enumerate(terminalFlagList):
                if item is False:
                    if dictList[index].get("terminalFlag") is not None:
                        if dictList[index]["terminalFlag"] is True:
                            terminalFlagList[index] = True
                            terminalCount += 1
                        else:
                            if dictList[index].get("currentProgress") is not None:
                                currentProgressStr[index] = dictList[index]["currentProgress"]

                nowTime = time.strftime("%Y-%m-%d %H:%M:%S")
                nowAllCurrentProgress = ""
                for currentProgressItem in currentProgressStr:
                    nowAllCurrentProgress += "%s\r\n" % (currentProgressItem)

            myLogger.myLogger_Logger().info(outputStr % (nowTime, nowAllCurrentProgress))


def experiment():
    # experimentInit()
    experimentMPMProcess = Process(target=experimentMultiProManager, name="experimentMultiProManager", args=())
    experimentMPMProcess.start()
    experimentMPMProcess.join()


if __name__ == "__main__":
    experiment()
