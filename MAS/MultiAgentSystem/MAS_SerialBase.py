#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : MAS_SerialBase.py
@Author  : jay.zhu
@Time    : 2022/11/6 15:01
"""
from MAS.MultiAgentSystem.MAS_Base import MAS_Base

class MAS_SerialBase(MAS_Base):
    def __init__(self, agents, masArgs, terminalHandler=None):
        super().__init__(agents, masArgs, terminalHandler)

    def optimizationInner(self):
        for agent in self.agents:
            agent.optimization()
            self.communication()