#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : MAS_Base.py
@Author  : jay.zhu
@Time    : 2022/10/12 22:29
"""

from EC.EC_Common import ArgsDictValueController

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
            if terminalHandler(agents = self.agents):
                return True

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
        for agent in self.agents:
            agent.update()

    def optimization(self):
        self.initShouldContinueOptimizationVar(self.terminalHandler)
        while self.shouldContinueOptimization(self.terminalHandler):
            self.optimizationInner()

    def optimizationInner(self):
        self.communication()
        for agent in self.agents:
            agent.optimization()

    def communication(self):
        pass





