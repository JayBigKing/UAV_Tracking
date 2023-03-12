#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_GPU_PredictAndSerialMAS.py
@Author  : jay.zhu
@Time    : 2023/3/3 23:53
"""
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_PredictAndNashMAS import UAV_MultiTarget_PredictAndNashMAS
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.GPU.UAV_MultiTarget_GPU_MAS_Base import UAV_MultiTarget_GPU_MAS_Base

class UAV_MultiTarget_GPU_PredictAndSerialMAS(UAV_MultiTarget_GPU_MAS_Base, UAV_MultiTarget_PredictAndNashMAS):
    def __init__(self, agents, masArgs, targetNum, terminalHandler=None, predictorCls=None, statRegisters=None,
                 deltaTime=1.):
        super().__init__(agents=agents,
                         masArgs=masArgs,
                         targetNum=targetNum,
                         terminalHandler=terminalHandler,
                         predictorCls=predictorCls,
                         statRegisters=statRegisters,
                         deltaTime=deltaTime)


    def clacDataForAimFuncSize(self, ):
        agentNum = len(self.agents)
        targetNum = self.targetNum

        targetRowLen = len(self.targetPositionList[0])
        uavRowLen = self.agents[0].optimizerInitArgs["optimizerLearningDimNum"]

        rowLen = uavRowLen if uavRowLen > targetRowLen else targetRowLen
        colLen = agentNum + targetNum

        return [rowLen, colLen]