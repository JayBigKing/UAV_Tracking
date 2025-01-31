#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : MAS_Base.py
@Author  : jay.zhu
@Time    : 2022/10/12 22:29
"""

from optimization.common.ArgsDictValueController import ArgsDictValueController

class MAS_Base:
    MAS_BASE_DEFAULT_ARGS = {
        "optimizationNeedTimes": 10
    }
    def __init__(self, agents, masArgs, terminalHandler=None):
        self.MAS_BASE_Args = ArgsDictValueController(userArgsDict=masArgs, defaultArgsDict=self.MAS_BASE_DEFAULT_ARGS)
        self.agents = agents
        self.terminalHandler = terminalHandler
        self._nowOptimizationTimeStep = 0


    def shouldContinueOptimization(self, terminalHandler):
        self._nowOptimizationTimeStep += 1
        if terminalHandler is not None:
            if terminalHandler(agents = self.agents) is False:
                return False


        if self._nowOptimizationTimeStep <= self.MAS_BASE_Args["optimizationNeedTimes"]:
            return True
        else:
            return False

    def initShouldContinueOptimizationVar(self, terminalHandler):
        if terminalHandler is not None:
            terminalHandler(initFlag=True)
        self._nowOptimizationTimeStep = 0

    def update(self):
        self.optimization()
        self.updateAgentState()

    def optimization(self):
        self.initShouldContinueOptimizationVar(self.terminalHandler)
        self.optimizationPreProcess()
        while self.shouldContinueOptimization(self.terminalHandler):
            self.optimizationInner()

    def optimizationInner(self):
        self.communication()
        for agent in self.agents:
            agent.optimization(init=True)

    def updateAgentState(self):
        '''
        agent take action, then update its state
        '''
        for agent in self.agents:
            agent.update()

    def optimizationPreProcess(self):
        pass

    def communication(self):
        pass

    def recvFromEnv(self, **kwargs):
        pass





