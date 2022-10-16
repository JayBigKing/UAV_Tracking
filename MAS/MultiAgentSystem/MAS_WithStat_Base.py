#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : MAS_WithStat_Base.py
@Author  : jay.zhu
@Time    : 2022/10/16 15:39
"""
from MAS.MultiAgentSystem.MAS_Base import MAS_Base
from dataStatistics.statFuncListGenerator import statFuncListGenerator

class MAS_WithStat_Base(MAS_Base):
    def __init__(self, agents, masArgs, terminalHandler=None, statRegisters=None):
        super().__init__(agents, masArgs, terminalHandler)

        self.__MAS_STAT_DEFAULT_FUNC_MAP = dict()

        self.statFuncReg = statFuncListGenerator(statRegisters, self.__MAS_STAT_DEFAULT_FUNC_MAP)

    def update(self):
        super().update()
        for item in self.statFuncReg:
            item()

