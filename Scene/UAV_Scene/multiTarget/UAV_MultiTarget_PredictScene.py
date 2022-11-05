#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : UAV_MultiTarget_PredictScene.py
@Author  : jay.zhu
@Time    : 2022/11/5 16:35
"""
from Scene.UAV_Scene.multiTarget.UAV_MultiTarget_Scene_Base import UAV_MultiTarget_Scene_Base

class UAV_MultiTarget_PredictScene(UAV_MultiTarget_Scene_Base):
    def __init__(self, agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                 MAS_Args, needRunningTime, predictorCls = None, targetNum=1, deltaTime=1., figureSavePath = None):
        MAS_Args["predictorCls"] = predictorCls
        super().__init__(agentsNum, agentsCls, agentsArgs, optimizerCls, optimizerArgs, targetCls, targetArgs, MAS_Cls,
                         MAS_Args, needRunningTime, targetNum, deltaTime, figureSavePath)

    def _initMAS(self, MAS_Cls, agents, MAS_Args, deltaTime):
        predictorCls = MAS_Args["predictorCls"]
        del MAS_Args["predictorCls"]
        self.multiAgentSystem = MAS_Cls(agents=agents,
                                        masArgs=MAS_Args,
                                        targetNum=self.targetNum,
                                        predictorCls = predictorCls,
                                        deltaTime=deltaTime)

    def runningInner(self):
        for item in self.targets:
            item.update()

        self.multiAgentSystem.recvFromEnv(targetPosition=[item.positionState for item in self.targets],
                                          targetVelocity=[item.velocity for item in self.targets])

        self.multiAgentSystem.update()