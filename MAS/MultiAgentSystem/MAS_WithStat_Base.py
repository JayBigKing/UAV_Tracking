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
    def __init__(self, agents, masArgs, terminalHandler=None, statRegisters=None, defaultStatFuncDict = None):
        super().__init__(agents, masArgs, terminalHandler)

        try:
            if defaultStatFuncDict is None:
                defaultStatFuncDict = dict()
            elif isinstance(defaultStatFuncDict, dict) is False:
                raise TypeError("The defaultStatFuncDict arg is not dict,"
                                "if you want to input a defaultStatFuncDict, please input a dict"
                                "or you can input nothing for defaultStatFuncDict")
        except TypeError as e:
            print(repr(e))

        self.statFuncReg = statFuncListGenerator(statRegisters, defaultStatFuncDict)

    def update(self):
        super().update()
        for item in self.statFuncReg:
            item()

