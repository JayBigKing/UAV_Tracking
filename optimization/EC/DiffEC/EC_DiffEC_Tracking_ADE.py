#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : EC_DiffEC_Tracking_ADE.py
@Author  : jay.zhu
@Time    : 2022/12/16 1:10
"""
from optimization.EC.DiffEC.EC_DiffEC_ADE import EC_DiffEC_ADE
from optimization.EC.DiffEC.EC_DiffEC_Tracking_Base import EC_DiffEC_Tracking_Base

class EC_DiffEC_Tracking_ADE(EC_DiffEC_ADE, EC_DiffEC_Tracking_Base):
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, otherTerminalHandler=None, useCuda=False):

        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                         statRegisters, otherTerminalHandler, useCuda)


