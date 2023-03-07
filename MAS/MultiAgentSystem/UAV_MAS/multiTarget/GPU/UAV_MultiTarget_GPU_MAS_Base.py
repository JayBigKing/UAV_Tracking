#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_GPU_MAS_Base.py
@Author  : jay.zhu
@Time    : 2023/3/3 23:41
"""
from numba import cuda
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_MAS_Base import UAV_MultiTarget_MAS_Base

class UAV_MultiTarget_GPU_MAS_Base(UAV_MultiTarget_MAS_Base):
    def __init__(self, agents, masArgs, targetNum, terminalHandler=None, statRegisters=None, deltaTime=1.):
        super().__init__(agents=agents,
                         masArgs=masArgs,
                         targetNum=targetNum,
                         terminalHandler=terminalHandler,
                         statRegisters=statRegisters,
                         deltaTime=deltaTime)

        self.updateDataForAimFunc()

    def updateDataForAimFunc(self, ):
        dataForAimFuncSize = self.clacDataForAimFuncSize()
        for item in self.agents:
            item.optimizer.ECArgsDictValueController["dataForAimFuncSize"] = dataForAimFuncSize
            item.optimizer.ECGPU_DataForAimFunc = cuda.device_array(dataForAimFuncSize)

    def clacDataForAimFuncSize(self, ):
        return 1

