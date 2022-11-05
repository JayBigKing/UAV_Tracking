#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : UAV_Scene_Base.py
@Author  : jay.zhu
@Time    : 2022/10/14 14:35
"""
from Scene.Scene_Base import Scene_Base
from Jay_Tool.visualizeTool.CoorDiagram import CoorDiagram
from dataStatistics.statFuncListGenerator import statFuncListGenerator

class UAV_Scene_Base(Scene_Base):
    def __init__(self, agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                 MAS_Args, needRunningTime, targetNum=1, deltaTime=1., figureSavePath = None, statOutputRegisters=None):
        self.agentsNum = agentsNum
        self.targetNum = targetNum
        self.deltaTime = deltaTime
        self.figureSavePath = figureSavePath
        self.__UAV_Scene_Base_stat_output_dict = {
            "UAV_SCENE_BASE_UAV_TRAJECTORY_VISUALIZE":self.UAV_SCENE_BASE_UAVTrajectoryVisualize
        }

        if statOutputRegisters is None:
            statOutputRegisters = ["UAV_SCENE_BASE_UAV_TRAJECTORY_VISUALIZE"]
        self.statOutputFuncReg = statFuncListGenerator(statOutputRegisters, self.__UAV_Scene_Base_stat_output_dict)

        self._initAgents(agentsCls, agentsArgs, optimizerCls, optimizerArgs, deltaTime)
        self._initTargets(targetCls, targetArgs, deltaTime)
        self._initMAS(MAS_Cls, self.agents, MAS_Args, deltaTime)

        super().__init__(self.agents, self.multiAgentSystem, needRunningTime)

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
        self.multiAgentSystem = MAS_Cls(agents, MAS_Args)

    def runningFinal(self):
        for item in self.statOutputFuncReg:
            item()

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
    def UAV_SCENE_BASE_SimpleVisualizeTrajectory(self, scattersList, nameList, titleName = None):
        cd = CoorDiagram()
        if self.figureSavePath is None:
            cd.drawManyScattersInOnePlane(scattersList, nameList=nameList, titleName=titleName)
        else:
            cd.setStorePath(self.figureSavePath)
            cd.drawManyScattersInOnePlane(scattersList, nameList=nameList, titleName=titleName, ifSaveFig=True)

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

        for i, item  in enumerate(self.agents):
            scattersList.append(item.coordinateVector)
            nameList.append(r"uav %d" % i)

        self.UAV_SCENE_BASE_SimpleVisualizeTrajectory(scattersList, nameList, titleName="uav trajectory")
