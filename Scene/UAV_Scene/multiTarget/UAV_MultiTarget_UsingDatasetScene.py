#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_UsingDatasetScene.py
@Author  : jay.zhu
@Time    : 2022/11/29 19:59
"""
from Jay_Tool.visualizeTool.CoorDiagram import CoorDiagram
from Scene.UAV_Scene.multiTarget.UAV_MultiTarget_PredictScene import UAV_MultiTarget_PredictScene


class UAV_MultiTarget_UsingDatasetScene(UAV_MultiTarget_PredictScene):
    # def __init__(self, agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
    #              MAS_Args, needRunningTime, predictorCls = None, targetNum=1, deltaTime=1., figureSavePath = None):
    def __init__(self, UAV_Dataset, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, MAS_Cls,
                 MAS_Args, needRunningTime, predictorCls=None, deltaTime=1., figureSavePath=None, userStatOutputRegisters = None,
                 sceneArgs=None):
        agentsNum, targetNum, targetArgs = self.UsingDataset_readUAVDataset(UAV_Dataset, agentsArgs, deltaTime)
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

    def _initTargets(self, targetCls, targetArgs, deltaTime):
        self.targets = [targetCls(initPositionState=targetArgs[i]["initPositionState"],
                                  targetTrajectory=targetArgs[i]["targetTrajectory"],
                                  deltaTime=deltaTime) for i in range(self.targetNum)]

    # def _initMAS(self, MAS_Cls, agents, MAS_Args, deltaTime):
    #     predictorCls = MAS_Args["predictorCls"]
    #     del MAS_Args["predictorCls"]
    #     self.multiAgentSystem = MAS_Cls(agents=agents,
    #                                     masArgs=MAS_Args,
    #                                     targetNum=self.targetNum,
    #                                     predictorCls=predictorCls,
    #                                     statRegisters=[ "recordNumOfTrackingUAVForTarget",
    #                                                     "recordDisOfUAVsForVisualize",
    #                                                     "recordAlertDisOfUAVsForVisualize",
    #                                                     "recordDisBetweenTargetAndUAV",
    #                                                     "recordEffectiveTime",
    #                                                     "recordFitness",
    #                                                     "recordTrackTargetID",
    #                                                     "recordDisBetweenCloseTar_UAV",],
    #                                     deltaTime=deltaTime)

    def UsingDataset_readUAVDataset(self, UAV_Dataset, agentsArgs, deltaTime=1.):
        agentsNum = UAV_Dataset["agentNum"]
        targetNum = UAV_Dataset["targetNum"]

        oneAgentInitArgs = agentsArgs["initArgs"]
        agentsArgs["initArgs"] = [{
            "initPositionState": [UAV_Dataset["agentInitPositionVec"][i][0],
                                  UAV_Dataset["agentInitPositionVec"][i][1], 0],
            "linearVelocityRange": oneAgentInitArgs["linearVelocityRange"],
            "angularVelocityRange": oneAgentInitArgs["angularVelocityRange"],
            "deltaTime": oneAgentInitArgs["deltaTime"],
        } for i in range(agentsNum)]

        targetArgs = [{
            "initPositionState": [UAV_Dataset["targetInitPositionVec"][i][0],
                                  UAV_Dataset["targetInitPositionVec"][i][1], 0],
            "linearVelocityRange": [0., 0.],
            "angularVelocityRange": [0., 0.],
            "movingFuncRegister": "randMoving",
            "deltaTime": deltaTime,
            "targetTrajectory": UAV_Dataset["targetTrajectories"][i]
        } for i in range(targetNum)]

        return agentsNum, targetNum, targetArgs
