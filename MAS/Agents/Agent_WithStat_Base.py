#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : Agent_WithStat_Base.py
@Author  : jay.zhu
@Time    : 2022/10/13 19:27
"""
from MAS.Agents.Agent_Base import Agent_Base
from dataStatistics.statFuncListGenerator import statFuncListGenerator

class Agent_WithStat_Base(Agent_Base):
    def __init__(self, optimizer = None, statRegisters=None):
        super().__init__(optimizer)

        self.__AGENT_STAT_DEFAULT_FUNC_MAP = dict()

        self.statFuncReg = statFuncListGenerator(statRegisters, self.__AGENT_STAT_DEFAULT_FUNC_MAP)

    def update(self):
        pass

    # def optimization(self):
    #     super().optimizer()
    #
    #     for item in self.statFuncReg:
    #         item(optimizationResult = self.optimizationResult)

