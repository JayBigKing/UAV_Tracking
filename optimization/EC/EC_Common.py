#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : EC_Common.py
@Author  : jay.zhu
@Time    : 2022/10/11 15:00
"""

# from copy import deepcopy
# from collections import defaultdict
# class ArgsDictValueController:
#     def __init__(self, userArgsDict, defaultArgsDict, onlyUseDefaultKey = False) -> None:
#         if isinstance(userArgsDict, defaultdict) is False:
#             self.userArgsDict = defaultdict(list)
#             if userArgsDict is not None:
#                 self.userArgsDict.update(dict(userArgsDict))
#         else:
#             self.userArgsDict = deepcopy(userArgsDict)
#
#         self.defaultArgsDict = dict(defaultArgsDict)
#         self.onlyUseDefaultKey = onlyUseDefaultKey
#
#         userArgsDictDelSet = set()
#         if self.onlyUseDefaultKey:
#             for key in self.userArgsDict.keys():
#                 if self.defaultArgsDict.get(key) is None:
#                     userArgsDictDelSet.add(key)
#             for key in userArgsDictDelSet:
#                 del self.userArgsDict[key]
#
#         for key in defaultArgsDict.keys():
#             if self.userArgsDict.get(key) is None:
#                 self.userArgsDict[key] = defaultArgsDict[key]
#
#
#     def getValueByKey(self, key : str) -> object:
#         # return self.userArgsDict[key] if self.userArgsDict.get(key) else self.defaultArgsDict[key]
#         return self.userArgsDict[key]
#
#     def setValueByKey(self, key : str, value : object) -> None:
#         if self.userArgsDict.get(key):
#             self.userArgsDict[key] = value
#         else:
#             raise KeyError("userArgsDict has no such key named %s " % key)
#
#     def __getitem__(self, item):
#         return self.getValueByKey(item)
#
#     def __setitem__(self, key, value):
#         self.setValueByKey(key, value)
#
#     # def update(self, userArgsDict : dict = None, defaultArgsDict : dict = None):
#     #     if userArgsDict is not None:
#     #         self.userArgsDict.update(userArgsDict)
#     #     if defaultArgsDict is not None:
#     #         self.defaultArgsDict.update(defaultArgsDict)
#     def update(self, newDict, onlyAddNotExists = True):
#         if onlyAddNotExists is True:
#             for key in newDict.keys():
#                 if self.userArgsDict.get(key) is None:
#                     self.userArgsDict[key] = newDict[key]
#         else:
#             self.userArgsDict.update(newDict)
#

