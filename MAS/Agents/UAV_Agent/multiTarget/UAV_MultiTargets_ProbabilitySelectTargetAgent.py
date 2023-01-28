#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTargets_ProbabilitySelectTargetAgent.py
@Author  : jay.zhu
@Time    : 2022/11/9 12:21
"""
import random
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_Agent import UAV_MultiTarget_Agent

class UAV_MultiTargets_ProbabilitySelectTargetAgent(UAV_MultiTarget_Agent):
    __UAV_MULTI_TARGET_PROBABILITY_SELECT_AGENT_DEFAULT_ARGS = {
        "selectByResProbability": .5,
    }

    def __init__(self, initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, targetNum, deltaTime=1., height=1., predictorCls=None,
                 predictorComputationArgs=None):
        super().__init__(initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, targetNum, deltaTime, height, predictorCls,
                 predictorComputationArgs)

        # 如果已经有了JBalanceFactor，由于只添加不存在的，所以使用用户输入的
        # 如果没有，就使用默认的
        self.agentArgs.update(newDict=self.__UAV_MULTI_TARGET_PROBABILITY_SELECT_AGENT_DEFAULT_ARGS, onlyAddNotExists=True)

    def optimization(self, **kwargs):
        if random.random() < self.agentArgs["selectByResProbability"]:
            self.trackLastTarget = True
        else:
            self.trackLastTarget = False
        super().optimization(**kwargs)

    def evalVars(self, chromosome):
        if self.trackLastTarget is True:
            chromosome[0] = float(self.trackingTargetIndex)
        return super().evalVars(chromosome)