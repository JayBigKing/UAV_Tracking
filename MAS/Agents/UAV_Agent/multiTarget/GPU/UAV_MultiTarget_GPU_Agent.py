#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_GPU_Agent.py
@Author  : jay.zhu
@Time    : 2023/2/26 21:41
"""
import numpy as np

from MAS.Agents.UAV_Agent.UAV_Common import calcMovingForUAV, calcDistance
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_Agent import UAV_MultiTarget_Agent
from numba import cuda


class UAV_MultiTarget_GPU_Agent(UAV_MultiTarget_Agent):
    def __init__(self, initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, targetNum, deltaTime=1., height=1., predictorCls=None,
                 predictorComputationArgs=None):

        super().__init__(initPositionState=initPositionState,
                         linearVelocityRange=linearVelocityRange,
                         angularVelocityRange=angularVelocityRange,
                         optimizerCls=optimizerCls,
                         agentArgs=agentArgs,
                         optimizerInitArgs=optimizerInitArgs,
                         optimizerComputationArgs=optimizerComputationArgs,
                         targetNum=targetNum,
                         deltaTime=deltaTime,
                         height=height,
                         predictorCls=predictorCls,
                         predictorComputationArgs=predictorComputationArgs)

    def recvMeg(self, **kwargs):
        super().recvMeg(**kwargs)
        dataForAimFunc = np.array(self.optimizer.ECArgsDictValueController["dataForAimFuncSize"])
        dataForAimFuncRowIndex = 0
        for i in range(len(self.numOfTrackingUAVForTargetList)):
            dataForAimFunc[dataForAimFuncRowIndex, i] = self.numOfTrackingUAVForTargetList[i]

        dataForAimFuncRowIndex += 1
        for i in range(self.targetNum):
            dataForAimFunc[dataForAimFuncRowIndex, :] = np.array(self.targetPositionList[i])
            dataForAimFuncRowIndex += 1

        dataForAimFuncRowIndex += 1
        for i in range(len(self.agentCrowd)):
            if i != self.selfIndex:
                dataForAimFunc[dataForAimFuncRowIndex, :] = np.array(self.agentCrowd["predictVelocityList"][i])
                dataForAimFuncRowIndex += 1

        self.optimizer.ECGPU_DataForAimFunc = cuda.device_array(dataForAimFunc)

