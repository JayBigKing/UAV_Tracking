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


class UAV_Scene_Base(Scene_Base):
    def __init__(self, agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                 MAS_Args, needRunningTime, targetNum=1, deltaTime=1.):
        self.agentsNum = agentsNum
        self.targetNum = targetNum
        self.deltaTime = deltaTime
        if isinstance(agentsArgs, list) is False:
            self.agents = [agentsCls(initPositionState=agentsArgs["initArgs"]["initPositionState"],
                                     linearVelocityRange=agentsArgs["initArgs"]["linearVelocityRange"],
                                     angularVelocity=agentsArgs["initArgs"]["angularVelocity"],
                                     agentArgs=agentsArgs["computationArgs"],
                                     optimizerCls=optimizerCls,
                                     optimizerInitArgs=optimizerArgs["optimizerInitArgs"],
                                     optimizerComputationArgs=optimizerArgs["optimizerComputationArgs"],
                                     deltaTime=deltaTime) for i in range(agentsNum)]
        else:
            self.agents = [agentsCls(initPositionState=agentsArgs[i]["initArgs"]["initPositionState"],
                                     linearVelocityRange=agentsArgs[i]["initArgs"]["linearVelocityRange"],
                                     angularVelocity=agentsArgs[i]["initArgs"]["angularVelocity"],
                                     agentArgs=agentsArgs[i]["computationArgs"],
                                     optimizerCls=optimizerCls,
                                     optimizerInitArgs=optimizerArgs["optimizerInitArgs"],
                                     optimizerComputationArgs=optimizerArgs["optimizerComputationArgs"],
                                     deltaTime=deltaTime) for i in range(agentsNum)]

        if self.targetNum == 1:
            self.targets = [targetCls(initPositionState=targetArgs["initPositionState"],
                                      linearVelocityRange=targetArgs["linearVelocityRange"],
                                      angularVelocity=targetArgs["angularVelocity"],
                                      movingFuncRegister=targetArgs["movingFuncRegister"],
                                      deltaTime=deltaTime)]
            self.target = self.targets[0]
        else:
            if isinstance(agentsArgs, list) is False:
                self.targets = [targetCls(initPositionState=targetArgs["initPositionState"],
                                          linearVelocityRange=targetArgs["linearVelocityRange"],
                                          angularVelocity=targetArgs["angularVelocity"],
                                          movingFuncRegister=targetArgs["movingFuncRegister"],
                                          deltaTime=deltaTime) for i in range(targetNum)]
            else:
                self.agents = [targetCls(initPositionState=targetArgs[i]["initPositionState"],
                                         linearVelocityRange=targetArgs[i]["linearVelocityRange"],
                                         angularVelocity=targetArgs[i]["angularVelocity"],
                                         movingFuncRegister=targetArgs[i]["movingFuncRegister"],
                                         deltaTime=deltaTime) for i in range(targetNum)]

        self.multiAgentSystem = MAS_Cls(self.agents, MAS_Args)
        super().__init__(self.agents, self.multiAgentSystem, needRunningTime)

    def runningFinal(self):
        scattersList = [self.target.coordinateVector]
        nameList = ["target"]
        for i, item  in enumerate(self.agents):
            scattersList.append(item.coordinateVector)
            nameList.append(r"uav %d" % i)
        cd = CoorDiagram()
        cd.drwaManyScattersInOnePlane(scattersList, nameList=nameList)

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
