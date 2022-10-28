#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : experimentForPredictTrajectory.py
@Author  : jay.zhu
@Time    : 2022/10/21 17:33
"""
import numpy as np
from algorithmTool.filterTool.KalmanFilter import KalmanFilter2
from Jay_Tool.visualizeTool.CoorDiagram import CoorDiagram

class TargetMovePredictor:
    def __init__(self, deltaTime=1):
        self.lastTargetX = np.zeros(2)
        self.predictTargetX = np.zeros(2)
        self.avgV = np.zeros(2)
        self.speedList = []
        self.deltaTime = deltaTime
        self.kalmanFiler = KalmanFilter2(R_val = 10. ,Q_val=.1)

        self.filterTimes = 0

    def predict(self, targetX, deltaTime = 1.):
        self.predictTargetX[0], self.predictTargetX[1] = self.kalmanFiler.process(targetX[0], targetX[1], deltaTime=deltaTime)
        return np.array(self.predictTargetX)

    def predictMultiSet(self, targetX, timeStepsQ=1., deltaTime = 1.):
        # predictTargetXList = np.zeros((timeStepsQ, 2))
        # predictTargetXList[0, 0], predictTargetXList[0, 1] = self.kalmanFiler.process(targetX[0], targetX[1],
        #                                                                               deltaTime=deltaTime)
        # for i in range(1, timeStepsQ):
        #     predictTargetXList[i, 0], predictTargetXList[i, 1] = self.kalmanFiler.process(predictTargetXList[i - 1, 0], predictTargetXList[i - 1, 1], deltaTime=deltaTime)
        #
        # return predictTargetXList
        return self.kalmanFiler.multiProcess(timeStepsQ, targetX[0], targetX[1], deltaTime=deltaTime)

def testKalmanOneByOne():
    SMAPLE_NUM_OF_TARGET = 120

    testTarget = [[i + 1 * np.random.random(), i*i] for i  in range(SMAPLE_NUM_OF_TARGET)]
    tmp = TargetMovePredictor(1)

    predictTargetXList = []
    for i in range(0, SMAPLE_NUM_OF_TARGET):
        predictTargetXList.append(tmp.predict(np.array(testTarget[i])))

    scattersList = [np.array(testTarget), np.array(predictTargetXList)]
    cd = CoorDiagram()
    cd.drwaManyScattersInOnePlane(scattersList)

def testKalmanMultiPoint():
    FIRST_SAMPLE_COUNT_FOR_KALMAN = 5
    PREDICT_STEPS = 5

    testTarget = [[i + 1 * np.random.random(), i*i ] for i  in range(200)]
    tmp = TargetMovePredictor(1)

    for i in range(0, FIRST_SAMPLE_COUNT_FOR_KALMAN):
        tmp.predict(np.array(testTarget[i]))

    predictTargetXList = tmp.predictMultiSet(testTarget[FIRST_SAMPLE_COUNT_FOR_KALMAN], PREDICT_STEPS)

    scattersList = [np.array(testTarget[FIRST_SAMPLE_COUNT_FOR_KALMAN:FIRST_SAMPLE_COUNT_FOR_KALMAN + PREDICT_STEPS]), np.array(predictTargetXList)]
    cd = CoorDiagram()
    cd.drwaManyScattersInOnePlane(scattersList)

def testKalmanMultiPointManyTimes():
    FIRST_SAMPLE_COUNT_FOR_KALMAN = 5
    PREDICT_STEPS = 5
    HOW_MANY_TIMES = 30
    nowTargetPointIndex = 0

    testTarget = [[i , i*i + 2 * np.random.random()] for i  in range(400)]
    tmp = TargetMovePredictor(1)
    for k in range(HOW_MANY_TIMES):

        for i in range(nowTargetPointIndex, FIRST_SAMPLE_COUNT_FOR_KALMAN):
            tmp.predict(np.array(testTarget[i]))

        startPredictPoint = nowTargetPointIndex + FIRST_SAMPLE_COUNT_FOR_KALMAN
        predictTargetXList = tmp.predictMultiSet(testTarget[startPredictPoint], PREDICT_STEPS)

        scattersList = [np.array(testTarget[startPredictPoint:startPredictPoint + PREDICT_STEPS]), np.array(predictTargetXList)]
        cd = CoorDiagram()
        cd.drwaManyScattersInOnePlane(scattersList)
        print(k)
        nowTargetPointIndex += FIRST_SAMPLE_COUNT_FOR_KALMAN

def testKalmanMultiPointManyTimesWithInitOnePredict():
    FIRST_SAMPLE_COUNT_FOR_KALMAN = 5
    INIT_ONE_PREDICT_COUNT = 10
    PREDICT_STEPS = 5
    HOW_MANY_TIMES = 10
    nowTargetPointIndex = INIT_ONE_PREDICT_COUNT

    testTarget = [[i , i*i + 2 * np.random.random()] for i  in range(400)]
    tmp = TargetMovePredictor(1)
    for k in range(INIT_ONE_PREDICT_COUNT):
        tmp.predict(np.array(testTarget[k]))
    for k in range(HOW_MANY_TIMES):

        for i in range(nowTargetPointIndex, FIRST_SAMPLE_COUNT_FOR_KALMAN):
            tmp.predict(np.array(testTarget[i]))

        startPredictPoint = nowTargetPointIndex + FIRST_SAMPLE_COUNT_FOR_KALMAN
        predictTargetXList = tmp.predictMultiSet(testTarget[startPredictPoint], PREDICT_STEPS)

        scattersList = [np.array(testTarget[startPredictPoint:startPredictPoint + PREDICT_STEPS]), np.array(predictTargetXList)]
        cd = CoorDiagram()
        cd.drwaManyScattersInOnePlane(scattersList)
        print(k)
        nowTargetPointIndex += FIRST_SAMPLE_COUNT_FOR_KALMAN

def main():
    testKalmanMultiPointManyTimesWithInitOnePredict()

if __name__ == "__main__":
    main()