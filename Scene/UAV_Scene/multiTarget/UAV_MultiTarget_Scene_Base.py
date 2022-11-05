#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : UAV_MultiTarget_Scene_Base.py
@Author  : jay.zhu
@Time    : 2022/11/5 14:06
"""
from Jay_Tool.visualizeTool.CoorDiagram import CoorDiagram
from Scene.UAV_Scene.UAV_Scene_Base import UAV_Scene_Base


class UAV_MultiTarget_Scene_Base(UAV_Scene_Base):
    def __init__(self, agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                 MAS_Args, needRunningTime, targetNum=1, deltaTime=1., figureSavePath=None):
        statOutputRegisters = [self.UAV_MULTITARGET_SCENE_BASE_TARGET_TRACKED_NUM_VISUALIZE,
                               self.UAV_MULTITARGET_SCENE_BASE_UAV_CONSUME_VISUALIZE,
                               "UAV_SCENE_BASE_UAV_TRAJECTORY_VISUALIZE"]
        super().__init__(agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                         MAS_Args, needRunningTime, targetNum, deltaTime, figureSavePath, statOutputRegisters)

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
                                      angularVelocity=targetArgs[i]["angularVelocityRange"],
                                      movingFuncRegister=targetArgs[i]["movingFuncRegister"],
                                      deltaTime=deltaTime) for i in range(self.targetNum)]

    def _initMAS(self, MAS_Cls, agents, MAS_Args, deltaTime):
        self.multiAgentSystem = MAS_Cls(agents=agents,
                                        MAS_Args=MAS_Args,
                                        targetNum=self.targetNum,
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
        if hasattr(self.multiAgentSystem, "numOfTrackingUAVForTargetStat"):
            numOfTrackingUAVForTargetStat = self.multiAgentSystem.numOfTrackingUAVForTargetStat
            for i, item in enumerate(numOfTrackingUAVForTargetStat):
                scattersList.append(item)
                nameList.append(r"target %d" % i)

            self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList, nameList, titleName="num of each target"
                                                                                            "tracked by uav")

    def UAV_MULTITARGET_SCENE_BASE_UAV_CONSUME_VISUALIZE(self):
        scattersList = []
        nameList = []
        if hasattr(self.multiAgentSystem, "consumeOfEachUAVStat"):
            consumeOfEachUAVStat = self.multiAgentSystem.consumeOfEachUAVStat
            for i, item in enumerate(consumeOfEachUAVStat):
                scattersList.append(item)
                nameList.append(r"uav %d" % i)

            self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList, nameList, titleName="each uav consume")
