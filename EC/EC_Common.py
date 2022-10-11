#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : EC_Common.py
@Author  : jay.zhu
@Time    : 2022/10/11 15:00
"""

from collections import defaultdict
class ArgsDictValueGetter:
    def __init__(self, userArgsDict : dict, defaultArgsDict : dict) -> None:
        if isinstance(userArgsDict, defaultdict) is False:
            self.userArgsDict = defaultdict(list)
            if userArgsDict is not None:
                self.userArgsDict.update(userArgsDict)
        else:
            self.userArgsDict = userArgsDict
        self.defaultArgsDict = defaultArgsDict
    def getValueByKey(self, key : str) -> object:
        return self.userArgsDict[key] if self.userArgsDict.get(key) else self.defaultArgsDict[key]
    def update(self, userArgsDict : dict = None, defaultArgsDict : dict = None):
        if userArgsDict is not None:
            self.userArgsDict.update(userArgsDict)
        if defaultArgsDict is not None:
            self.defaultArgsDict.update(defaultArgsDict)

