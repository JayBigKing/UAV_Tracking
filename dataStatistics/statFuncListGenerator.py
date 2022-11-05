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
    statFuncReg = set()
    if statRegisters is not None:
        for item in statRegisters:
            if defaultStatFuncDict is not None and isinstance(item, str):
                statFuncReg.add(defaultStatFuncDict[item])
            else:
                statFuncReg.add(item)

    return statFuncReg

def statFuncListAdder(statFuncReg: set, appendFunc) -> None:
    try:
        if isfunction(appendFunc):
            if appendFunc not in statFuncReg:
                statFuncReg.add(appendFunc)
        else:
            raise TypeError("The second input args should be a function"
                            "Please give a function which you want to add, "
                            "while the function is also not in the register list")
    except TypeError as e:
        print(repr(e))

def statFuncListAdderByRegister(statFuncReg: set, statRegisters, defaultStatFuncDict):
    try:
        if isinstance(statRegisters, set):
            if statRegisters is not None:
                for item in statRegisters:
                    if defaultStatFuncDict is not None and isinstance(item, str):
                        statFuncReg.add(defaultStatFuncDict[item])
                    else:
                        statFuncReg.add(item)
        else:
            raise TypeError("statFuncReg is expected a set instance")
    except TypeError as e:
        print(repr(e))

def statFuncListDeleter(statFuncReg: set, deleteFunc) -> None:
    try:
        if isfunction(deleteFunc):
            if deleteFunc in statFuncReg:
                statFuncReg.remove(deleteFunc)
        else:
            raise TypeError("The second input args should be a function"
                            "Please give a function which you want to delete, "
                            "while the function is also in the register list")
    except TypeError as e:
        print(repr(e))