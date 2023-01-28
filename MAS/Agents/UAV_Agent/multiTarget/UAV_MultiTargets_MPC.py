#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTargets_MPC.py
@Author  : jay.zhu
@Time    : 2023/1/27 14:42
"""
import numpy as np
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_ProbabilitySelectTargetAgent import \
    UAV_MultiTargets_ProbabilitySelectTargetAgent


class UAV_MultiTargets_MPC(UAV_MultiTargets_ProbabilitySelectTargetAgent):
    def __init__(self, initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                 optimizerInitArgs, optimizerComputationArgs, targetNum, deltaTime=1., height=1., predictorCls=None,
                 predictorComputationArgs=None):
        super().__init__(initPositionState, linearVelocityRange, angularVelocityRange, optimizerCls, agentArgs,
                         optimizerInitArgs, optimizerComputationArgs, targetNum, deltaTime, height, predictorCls,
                         predictorComputationArgs)
        self.firstMPC = True

    def optimization(self, **kwargs):
        if self.remainMoving == 0:
            if kwargs.get("init"):
                if self.firstMPC is True:
                    self.firstMPC = False
                else:
                    self.adjustBestSolutionMPC()
        super().optimization()

    def adjustBestSolutionMPC(self):
        newSolutionMPC = np.array(self.optimizer.bestChromosome[:, self.optimizer.BEST_IN_ALL_GEN_DIM_INDEX])
        step = 0
        for i in range(self.agentArgs["usePredictVelocityLen"],
                       self.agentArgs["predictVelocityLen"]):
            if step == self.agentArgs["predictVelocityLen"] - 1:
                break
            adjustStartIndex, adjustEndIndex = self.getVelocityFromPredictVelocityList(
                step)
            useStartIndex, useEndIndex = self.getVelocityFromPredictVelocityList(
                i)
            newSolutionMPC[adjustStartIndex: adjustEndIndex] = newSolutionMPC[useStartIndex: useEndIndex]
            step += 1

        newSolutionMPCFitness, newSolutionMPCAim = self.optimizer.fittingOne(
            newSolutionMPC, self.optimizer.evalVars)
        if self.optimizer.cmpFitting(newSolutionMPCFitness, self.optimizer.bestChromosomesFittingValue[
            self.optimizer.BEST_IN_ALL_GEN_DIM_INDEX]) > 0:
            self.optimizer.bestChromosome[:, self.optimizer.BEST_IN_ALL_GEN_DIM_INDEX] = np.array(newSolutionMPC)
            self.optimizer.bestChromosomesFittingValue[self.optimizer.BEST_IN_ALL_GEN_DIM_INDEX], \
            self.optimizer.bestChromosomesAimFuncValue[
                self.optimizer.BEST_IN_ALL_GEN_DIM_INDEX] = newSolutionMPCFitness, newSolutionMPCAim
