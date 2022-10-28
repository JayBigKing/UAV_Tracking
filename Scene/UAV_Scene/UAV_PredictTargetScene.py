#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : UAV_PredictTargetScene.py
@Author  : jay.zhu
@Time    : 2022/10/28 20:57
"""
from Scene.UAV_Scene.UAV_Scene_Base import UAV_Scene_Base

class UAV_PredictTargetScene(UAV_Scene_Base):
    def __init__(self, agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                 MAS_Args, needRunningTime, predictorCls = None, targetNum=1, deltaTime=1.):
        MAS_Args["predictorCls"] = predictorCls
        super().__init__(agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                 MAS_Args, needRunningTime, targetNum, deltaTime)


    def _initMAS(self, MAS_Cls, agents, MAS_Args, deltaTime):
        predictorCls = MAS_Args["predictorCls"]
        del MAS_Args["predictorCls"]
        self.multiAgentSystem = MAS_Cls(agents, MAS_Args, predictorCls, deltaTime)

    def runningInner(self):
        if self.targetNum == 1:
            self.target.update()
        else:
            for item in self.targets:
                item.update()

        if self.targetNum == 1:
            self.multiAgentSystem.recvFromEnv(targetPosition=self.target.positionState, targetVelocity=self.target.velocity)
        else:
            self.multiAgentSystem.recvFromEnv(targetPosition=[item.positionState for item in self.targets],
                                              targetVelocity=[item.velocity for item in self.targets])

        # self.multiAgentSystem.optimization()
        self.multiAgentSystem.update()



