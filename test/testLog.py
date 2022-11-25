#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : testLog.py
@Author  : jay.zhu
@Time    : 2022/11/14 16:37
"""
from Jay_Tool.LogTool import myLogger
myLogger.myLogger_Init()
# myLogger.consoleLogger.info("hello world")
# myLogger.myLogger_ConsoleLogger().info("hello world")
# myLogger.myLogger_FileLogger().info("hello world")
myLogger.myLogger_Logger().info("hello world")