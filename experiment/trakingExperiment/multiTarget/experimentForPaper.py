#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : experimentForPaper.py
@Author  : jay.zhu
@Time    : 2023/1/18 16:51
"""
import sys
import os

sys.path.append("../../../")
from experiment.experimentInit import experimentInit
from Jay_Tool.LogTool import myLogger
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_Agent import UAV_MultiTarget_Agent
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_ProbabilitySelectTargetAgent import \
    UAV_MultiTargets_ProbabilitySelectTargetAgent
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_MPC import UAV_MultiTargets_MPC
import experimentBase
saveFigPathPrefix = "../../experimentRes/experimentForPaper/"

# DATASET_PATH = "targetTrajectoryDataset/randTarget2.json"

experimentBase.NEED_RUNNING_TIME = 200
@clockTester
def experiment1():
    '''
    定性分析，获取跟踪图，和target的平均距离，UAV之间的平均距离，跟踪target的UAV数量
    '''
    movingWayList = [["straight", "movingStraightly"], ["sin", "movingAsSin"], ["rand", "randMoving"]]
    for item in movingWayList:
        figureSavePathKeyword = "%sexperiment1_%s" % (saveFigPathPrefix, item[0])
        myLogger.myLogger_Logger().info("experiment1 : %s start" % item[0])
        uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                                  getTargetTrajectoryWay="online",
                                                  masKey="PredictAndNashMAS",
                                                  optimizerKey="dynEC",
                                                  agentsNum=4,
                                                  targetNum=2,
                                                  targetMovingWay = item[1],
                                                  figureSavePathKeyword=figureSavePathKeyword,
                                                  userStatOutputRegisters=[])
        uav_scene.run()

    # figureSavePathKeyword = "%s%s" % (saveFigPathPrefix, "experiment1_sin")
    # uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
    #                                           getTargetTrajectoryWay="online",
    #                                           masKey="PredictAndNashMAS",
    #                                           optimizerKey="dynEC",
    #                                           agentsNum=4,
    #                                           targetNum=2,
    #                                           targetMovingWay = "movingAsSin",
    #                                           figureSavePathKeyword=figureSavePathKeyword,
    #                                           userStatOutputRegisters=[])
    # uav_scene.run()
    #
    # figureSavePathKeyword = "%s%s" % (saveFigPathPrefix, "experiment1_rand")
    # uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
    #                                           getTargetTrajectoryWay="online",
    #                                           masKey="PredictAndNashMAS",
    #                                           optimizerKey="dynEC",
    #                                           agentsNum=4,
    #                                           targetNum=2,
    #                                           targetMovingWay = "randMoving",
    #                                           figureSavePathKeyword=figureSavePathKeyword,
    #                                           userStatOutputRegisters=[])
    # uav_scene.run()

@clockTester
def experiment2():
    '''
    定量分析，是否平衡追踪一台target的uav数量，有效跟踪占比，平均有效跟踪时间
    '''

    def experiment2Inner(agentsNum=4, targetNum=2, targetMovingWay = "randMoving", runningTimes = 1):
        for i in range(runningTimes):
            newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
            figureSavePathKeyword = "%s%s_a%d_t%d" % (saveFigPathPrefix, "experiment2_balance", agentsNum, targetNum)
            myLogger.myLogger_Logger().info("%s_a%d_t%d start,      %d / %d" % ("experiment2_balance", agentsNum, targetNum, i+1, runningTimes))
            uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                                      getTargetTrajectoryWay="dataset",
                                                      masKey="PredictAndNashMAS",
                                                      optimizerKey="dynEC",
                                                      datasetPath=newDatasetPath,
                                                      targetMovingWay = "randMoving",
                                                      figureSavePathKeyword=figureSavePathKeyword,
                                                      userStatOutputRegisters=["UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE",
                                                          "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_FOR_TARGET_STORE"])
            uav_scene.run()

            figureSavePathKeyword = "%s%s_a%d_t%d" % (saveFigPathPrefix, "experiment2_no_balance", agentsNum, targetNum)
            myLogger.myLogger_Logger().info("%s_a%d_t%d start,      %d / %d" % ("experiment2_no_balance", agentsNum, targetNum, i+1, runningTimes))
            agentComputationArgs = dict(experimentBase.S_DEFAULT_AGENT_COMPUTATION_ARGS)
            agentComputationArgs["JBalanceFactor"] = 0.

            uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                                      getTargetTrajectoryWay="dataset",
                                                      masKey="PredictAndNashMAS",
                                                      optimizerKey="dynEC",
                                                      datasetPath=newDatasetPath,
                                                      targetMovingWay = "randMoving",
                                                      agentComputationArgs=agentComputationArgs,
                                                      figureSavePathKeyword=figureSavePathKeyword,
                                                      userStatOutputRegisters=["UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE",
                                                          "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_FOR_TARGET_STORE"])
            uav_scene.run()

            os.remove(newDatasetPath)

    experiment2RunningTimes = 3
    experiment2Inner(agentsNum=4,
                     targetNum=5,
                     runningTimes=experiment2RunningTimes)
    experiment2Inner(agentsNum=5,
                     targetNum=5,
                     runningTimes=experiment2RunningTimes)
    experiment2Inner(agentsNum=6,
                     targetNum=5,
                     runningTimes=experiment2RunningTimes)

@clockTester
def experiment3():
    '''
    定量分析，是否使用纳什优化的uav之间距离，alert等
    '''
    agentsNum = 4
    targetNum = 2
    targetMovingWay = "randMoving"
    newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
    useNashList = [["nash", "PredictAndNashMAS"], ["noNash", "PredictMAS"]]
    for item in useNashList:
        figureSavePathKeyword = "%sexperiment3_%s" % (saveFigPathPrefix, item[0])
        myLogger.myLogger_Logger().info("experiment3 : %s start" % item[0])
        uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                                  getTargetTrajectoryWay="dataset",
                                                  masKey=item[1],
                                                  optimizerKey="dynEC",
                                                  datasetPath=newDatasetPath,
                                                  targetMovingWay="randMoving",
                                                  figureSavePathKeyword=figureSavePathKeyword,
                                                  userStatOutputRegisters=[
                                                      "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STORE",
                                                      "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE",
                                                      "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore"
                                                        ])
        uav_scene.run()
    os.remove(newDatasetPath)

@clockTester
def experiment4():
    '''
    定量分析，是否使用DEO的与uav的平均距离，fitness，稳定性
    '''
    agentsNum = 4
    targetNum = 2
    targetMovingWay = "randMoving"
    newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
    experimentOptimizerList = ["dynEC", "noDynEC", "DE", "PSO"]
    for item in experimentOptimizerList:
        figureSavePathKeyword = "%sexperiment4_%s" % (saveFigPathPrefix, item)
        myLogger.myLogger_Logger().info("experiment4 : %s start" % item)
        uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                                  getTargetTrajectoryWay="dataset",
                                                  masKey="PredictAndNashMAS",
                                                  optimizerKey=item,
                                                  datasetPath=newDatasetPath,
                                                  targetMovingWay="randMoving",
                                                  figureSavePathKeyword=figureSavePathKeyword,
                                                  userStatOutputRegisters=[
                                                      "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STORE",
                                                      "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE",
                                                      "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore"
                                                        ])
        uav_scene.run()
    os.remove(newDatasetPath)

@clockTester
def experiment5():
    '''
    定量分析，是否使用概率型agent
    '''
    agentsNum = 4
    targetNum = 2
    targetMovingWay = "randMoving"
    newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
    ifUseProList = [["pro", UAV_MultiTargets_ProbabilitySelectTargetAgent], ["noPro", UAV_MultiTarget_Agent]]
    for item in ifUseProList:
        figureSavePathKeyword = "%sexperiment5_%s" % (saveFigPathPrefix, item[0])
        myLogger.myLogger_Logger().info("experiment5 : %s start" % item[0])
        uav_scene = experimentBase.experimentBase(agentCls=item[1],
                                                  getTargetTrajectoryWay="dataset",
                                                  masKey="PredictAndNashMAS",
                                                  optimizerKey="dynEC",
                                                  datasetPath=newDatasetPath,
                                                  targetMovingWay="randMoving",
                                                  figureSavePathKeyword=figureSavePathKeyword,
                                                  userStatOutputRegisters=[
                                                      "UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_STORE"
                                                        ])
        uav_scene.run()
    os.remove(newDatasetPath)

@clockTester
def experiment6():
    '''
    定量分析，是否使用概率型agent
    '''
    agentsNum = 4
    targetNum = 2
    targetMovingWay = "randMoving"
    newDatasetPath = experimentBase.generateDataset(agentsNum, targetNum, targetMovingWay)
    ifUseProList = [["MPC", UAV_MultiTargets_MPC], ["noMPC", UAV_MultiTargets_ProbabilitySelectTargetAgent]]
    for item in ifUseProList:
        figureSavePathKeyword = "%sexperiment6_%s" % (saveFigPathPrefix, item[0])
        myLogger.myLogger_Logger().info("experiment6 : %s start" % item[0])
        uav_scene = experimentBase.experimentBase(agentCls=item[1],
                                                  getTargetTrajectoryWay="dataset",
                                                  masKey="PredictAndNashMAS",
                                                  optimizerKey="dynEC",
                                                  datasetPath=newDatasetPath,
                                                  targetMovingWay="randMoving",
                                                  figureSavePathKeyword=figureSavePathKeyword,
                                                  userStatOutputRegisters=[
                                                      "UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_STORE"
                                                        ])
        uav_scene.run()
    os.remove(newDatasetPath)

@clockTester
def experimentPSO():
    figureSavePathKeyword = "%s%s" % (saveFigPathPrefix, "experimentPSO")
    myLogger.myLogger_Logger().info("%s start" % ("PSO"))
    uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                              getTargetTrajectoryWay="online",
                                              masKey="PredictAndNashMAS",
                                              optimizerKey="PSO",
                                              agentsNum=4,
                                              targetNum=2,
                                              targetMovingWay = "movingStraightly",
                                              figureSavePathKeyword=figureSavePathKeyword,
                                              userStatOutputRegisters=[])
    uav_scene.run()

@clockTester
def experimentNoDyn():
    figureSavePathKeyword = "%s%s" % (saveFigPathPrefix, "experimentnoDynEC")
    myLogger.myLogger_Logger().info("%s start" % ("noDynEC"))
    uav_scene = experimentBase.experimentBase(agentCls=UAV_MultiTargets_ProbabilitySelectTargetAgent,
                                              getTargetTrajectoryWay="online",
                                              masKey="PredictMAS",
                                              optimizerKey="noDynEC",
                                              agentsNum=4,
                                              targetNum=2,
                                              targetMovingWay = "movingStraightly",
                                              figureSavePathKeyword=figureSavePathKeyword,
                                              userStatOutputRegisters=[])
    uav_scene.run()

def experiment():
    experimentInit()
    # experiment1()
    # experiment2()
    # experiment3()
    # experiment4()
    # experiment5()
    # experimentPSO()
    experiment6()


if __name__ == "__main__":
    experiment()
