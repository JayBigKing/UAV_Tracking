#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : UAV_Agent_MultiTarget.py
@Author  : jay.zhu
@Time    : 2022/10/21 17:24
"""
from MAS.Agents.UAV_Agent.UAV_Common import calcMovingForUAV, calcDistance
from MAS.Agents.UAV_Agent.UAV_Agent import UAV_Agent


class UAV_Agent_MultiTarget(UAV_Agent):
    __UAV_AGENT_MULTI_TARGET_DEFAULT_ARGS = {
        "predictVelocityLen": 1,
        "usePredictVelocityLen": 1,
        "sameBestFittingCountThreshold": 10,
        "fittingIsSameThreshold": 1e-4,
        "JTaskFactor": 1.,
        "JConFactor": 1.,
        "JColFactor": 1.,
        "JComFactor": 1.,
        "minDistanceThreshold": 10.,
        "smallDistanceBlameFactor": 19.,
        "maxDistanceThreshold": 30.,
        "bigDistanceBlameFactor": 15.,
    }

    def __init__(self, initPositionState, linearVelocityRange, angularVelocity, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, deltaTime=1., height=1., predictorCls=None,
                 predictorComputationArgs=None):
        pass
