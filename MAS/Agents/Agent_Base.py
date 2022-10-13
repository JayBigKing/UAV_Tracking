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

    def sendMeg(self):
        pass

    def recvMeg(self, **kwargs):
        pass

    def optimization(self):
        self.optimizationResult = self.optimizer.optimization()

    @abstractmethod
    def update(self):
        pass

