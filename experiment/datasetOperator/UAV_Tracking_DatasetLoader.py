#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_Tracking_DatasetLoader.py
@Author  : jay.zhu
@Time    : 2022/11/26 17:40
"""
import json

class UAV_Tracking_DatasetLoader:
    def __init__(self):
        pass

    @staticmethod
    def loadJson(fileName):
        with open(fileName, 'r') as f:
            jsonData = json.load(f)
        return jsonData

    def loadDataset(self, fileName):
        return self.loadJson(fileName)




