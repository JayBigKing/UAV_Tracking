#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : statFuncListGenerator.py
@Author  : jay.zhu
@Time    : 2022/10/13 19:51
"""
from inspect import isfunction

def statFuncListGenerator(statRegisters, defaultStatFuncDict):
    statFuncReg = []
    for item in statRegisters:
        if isinstance(item, str):
            statFuncReg.append(defaultStatFuncDict[item])
        else:
            statFuncReg.append(item)

    return statFuncReg

