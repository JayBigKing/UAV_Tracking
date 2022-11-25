#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : Scene_Base.py
@Author  : jay.zhu
@Time    : 2022/10/14 14:29
"""
from Jay_Tool.LogTool import myLogger


class Scene_Base:
    def __init__(self, agents, multiAgentSystem, needRunningTimes, ):
        self.agents = agents
        self.multiAgentSystem = multiAgentSystem
        self.needRunningTimes = needRunningTimes
        self.nowRunningTime = 0

    def shouldContinue(self):
        if self.nowRunningTime < self.needRunningTimes:
            self.nowRunningTime += 1
            return True
        else:
            return False

    def runningPreProcess(self):
        pass

    def runningInner(self):
        pass

    def runningFinal(self):
        pass

    def run(self):
        self.runningPreProcess()
        while self.shouldContinue():
            self.runningInner()
            myLogger.myLogger_Logger().info('running epoch : %d / %d' % (self.nowRunningTime, self.needRunningTimes))
        self.runningFinal()



