#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : testTarget.py
@Author  : jay.zhu
@Time    : 2022/10/13 20:54
"""
import numpy as np
from Jay_Tool.visualizeTool.CoorDiagram import  CoorDiagram
from MAS.Agents.UAV_Agent.UAV_TargetAgent import UAV_TargetAgent

def test1():
    target0 = UAV_TargetAgent([0., 0., 0.], [0., 10.], [0., 20.], movingFuncRegister="randMoving", deltaTime=.1 )
    for i in range(400):
        # target0.optimization()
        target0.update()

    cd = CoorDiagram()
    scattersList = [target0.coordinateVector]
    cd.drawManyScattersInOnePlane(scattersList, ifSaveFig=True, nameList=["hhh"])

def main():
    test1()

if __name__ == "__main__":
    main()
