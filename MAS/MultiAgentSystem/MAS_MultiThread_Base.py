#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : MAS_MultiThread_Base.py
@Author  : jay.zhu
@Time    : 2022/12/17 21:59
"""
from MAS.MultiAgentSystem.MAS_Base import MAS_Base
from multiprocessing import Process

class MAS_MultiThread_Base(MAS_Base):
    def callAgentProcess(self, agent):
        agent.optimization()
    def __init__(self, agents, masArgs, terminalHandler=None):
        super().__init__(agents, masArgs, terminalHandler)

    def optimizationInner(self):
        self.communication()

        process_list = [Process(target=self.callAgentProcess, args=(agent,)) for agent in self.agents]
        for p in process_list:
            p.start()
        for p in process_list:
            p.join()
