#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : experimentGPU.py
@Author  : jay.zhu
@Time    : 2023/2/26 21:35
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
import experimentBase

saveFigPathPrefix = "../../experimentRes/experimentForPaper/"
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
