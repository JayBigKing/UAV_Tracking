#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : Agent_Base.py
@Author  : jay.zhu
@Time    : 2022/10/13 12:18
"""
from abc import ABC, abstractmethod

class Agent_Base(ABC):
    def __init__(self, optimizer = None):
        self.optimizer = optimizer
        self.optimizationResult = None
        self.firstRun = True

    def sendMeg(self):
        pass

    def recvMeg(self, **kwargs):
        pass

    def optimization(self, **kwargs):
        self.firstRun = False
        self.optimizationResult = self.optimizer.optimize(**kwargs)

    @abstractmethod
    def update(self):
        pass

