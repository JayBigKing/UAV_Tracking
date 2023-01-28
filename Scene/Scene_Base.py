#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : Scene_Base.py
@Author  : jay.zhu
@Time    : 2022/10/14 14:29
"""
from Jay_Tool.LogTool import myLogger
from optimization.common.ArgsDictValueController import ArgsDictValueController

class Scene_Base:
    SCENE_BASE_DEFAULT_ARGS = {
        "ifPrintRunningEpoch" : True
    }
    def __init__(self, agents, multiAgentSystem, needRunningTimes, sceneArgs = None):
        self.agents = agents
        self.multiAgentSystem = multiAgentSystem
        self.needRunningTimes = needRunningTimes
        self.nowRunningTime = 0
        if sceneArgs is not None:
            self.SCENE_BASE_Args = ArgsDictValueController(userArgsDict=sceneArgs,
                                                           defaultArgsDict=self.SCENE_BASE_DEFAULT_ARGS)
        else:
            self.SCENE_BASE_Args = ArgsDictValueController(userArgsDict=dict(),
                                                           defaultArgsDict=self.SCENE_BASE_DEFAULT_ARGS)

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
            if self.SCENE_BASE_Args["ifPrintRunningEpoch"] is True:
                myLogger.myLogger_Logger().info('running epoch : %d / %d' % (self.nowRunningTime, self.needRunningTimes))
        self.runningFinal()



