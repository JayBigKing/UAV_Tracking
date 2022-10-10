import numpy as np
from copy import deepcopy
import sys
sys.path.append("../algorithmTool/filterTool")
sys.path.append("../visualize")
from CoorDiagram import CoorDiagram
from KalmanFilter import KalmanFilter2

X_DEFAULT_Q_2D_MAT = np.mat([[0.001, 0.], [0., 0.001]])
X_DEFAULT_R_2D_MAT = np.mat([[10., 0.], [0., 10.]])
Y_DEFAULT_Q_2D_MAT = np.mat([[0.001, 0.], [0., 0.001]])
Y_DEFAULT_R_2D_MAT = np.mat([[10., 0.], [0., 10.]])


class TargetMovePredictor:
    def __init__(self, deltaTime=1):
        self.lastTargetX = np.zeros(2)
        self.predictTargetX = np.zeros(2)
        self.avgV = np.zeros(2)
        self.speedList = []
        self.deltaTime = deltaTime
        self.kalmanFiler = KalmanFilter2()

        self.filterTimes = 0

    def predict(self, targetX, deltaTime = 1):
        self.predictTargetX[0], self.predictTargetX[1] = self.kalmanFiler.process(targetX[0], targetX[1], deltaTime=deltaTime)
        return deepcopy(self.predictTargetX)

    def predictMultiSet(self, targetX, timeStepsQ=1, deltaTime = 1):
        # predictTargetXList = np.zeros((timeStepsQ, 2))
        # predictTargetXList[0, 0], predictTargetXList[0, 1] = self.kalmanFiler.process(targetX[0], targetX[1],
        #                                                                               deltaTime=deltaTime)
        # for i in range(1, timeStepsQ):
        #     predictTargetXList[i, 0], predictTargetXList[i, 1] = self.kalmanFiler.process(predictTargetXList[i - 1, 0], predictTargetXList[i - 1, 1], deltaTime=deltaTime)
        #
        # return predictTargetXList
        return self.kalmanFiler.multiProcess(timeStepsQ, targetX[0], targetX[1], deltaTime=deltaTime)

# class TargetMovePredictor:
#     def __init__(self, deltaTime=1):
#         self.lastTargetX = np.zeros(2)
#         self.predictTargetX = np.zeros(2)
#         self.avgV = np.zeros(2)
#         self.speedList = []
#         self.deltaTime = deltaTime
#         self.xKalmanFiler = KalmanFilter()
#         self.yKalmanFiler = KalmanFilter()
#
#         self.xKalmanFiler.init2D(qMat=X_DEFAULT_Q_2D_MAT,
#                                  rMat=X_DEFAULT_R_2D_MAT, timeInval=deltaTime)
#         self.yKalmanFiler.init2D(qMat=Y_DEFAULT_Q_2D_MAT,
#                                  rMat=Y_DEFAULT_R_2D_MAT, timeInval=deltaTime)
#
#         self.filterTimes = 0
#
#     def predict(self, targetX):
#         self.filterTimes += 1
#         self.speedList.append(
#             [(targetX[0] - self.lastTargetX[0]) / self.deltaTime, (targetX[1] - self.lastTargetX[1]) / self.deltaTime])
#         if self.filterTimes == 1:
#             self.speedList[0][0] = self.speedList[0][1] = 1.
#         self.predictTargetX[0], _ = self.xKalmanFiler.kalmanFilterCalc2D(targetX[0], self.speedList[-1][0])
#         self.predictTargetX[1], _ = self.yKalmanFiler.kalmanFilterCalc2D(targetX[1], self.speedList[-1][1])
#         self.lastTargetX[0], self.lastTargetX[1] = targetX[0], targetX[1]
#         return deepcopy(self.predictTargetX)
#
#     def predictMultiSte(self, targetX, timeStepsQ=1):
#         self.filterTimes += 1
#         predictTargetXList = np.zeros((timeStepsQ, 2))
#         lastTargetX = np.array(self.lastTargetX)
#         lastTargetX[0], lastTargetX[1] = self.lastTargetX[0], self.lastTargetX[1]
#         predictV = np.array(
#             [(targetX[0] - self.lastTargetX[0]) / self.deltaTime, (targetX[1] - self.lastTargetX[1]) / self.deltaTime])
#         self.speedList.append(predictV)
#         if self.filterTimes == 1:
#             self.speedList[0][0] = self.speedList[0][1] = predictV[0] = predictV[1] =  1.
#         # if self.filterTimes > 10:
#         #     if predictV[0] < 1e-2 and predictV[1] < 1e-2:
#         #         predictV = np.average(self.speedList, axis=1)
#         #     else:
#         #         predictV = 0.5 * (predictV + np.average(self.speedList, axis=0))
#         predictTargetXList[0, 0]  = targetX[0]
#         predictTargetXList[0, 1]  = targetX[1]
#         # predictTargetXList[0, 0], _ = self.xKalmanFiler.kalmanFilterCalc2D(targetX[0], predictV[0])
#         # predictTargetXList[0, 1], _ = self.yKalmanFiler.kalmanFilterCalc2D(targetX[1], predictV[1])
#
#         for i in range(1, timeStepsQ):
#             predictTargetXList[i, 0], _ = self.xKalmanFiler.kalmanFilterCalc2D(predictTargetXList[i - 1, 0],
#                                                                                predictV[0])
#             predictTargetXList[i, 1], _ = self.yKalmanFiler.kalmanFilterCalc2D(predictTargetXList[i - 1, 1],
#                                                                                predictV[1])
#             # lastTargetX[0], lastTargetX[1] = targetX[0], targetX[1]
#             if i < timeStepsQ - 1:
#                 predictTargetXList[i + 1, 0], predictTargetXList[i + 1, 1] = predictTargetXList[i, 0] + predictV[
#                     0] * self.deltaTime, predictTargetXList[i, 1] + predictV[1] * self.deltaTime
#
#         self.lastTargetX[0], self.lastTargetX[1] = targetX[0], targetX[1]
#
#         return predictTargetXList

        # predictV = np.array(
        #     [(targetX[0] - self.lastTargetX[0]) / self.deltaTime, (targetX[1] - self.lastTargetX[1]) / self.deltaTime])
        # predictTargetXList = np.zeros((timeStepsQ, 2))
        # xRes = self.xKalmanFiler.kalmanFilterCalc2DMulti(targetX[0],predictV[0], timeStepsQ)
        # yRes = self.yKalmanFiler.kalmanFilterCalc2DMulti(targetX[1], predictV[1], timeStepsQ)
        # for i in range(timeStepsQ):
        #     predictTargetXList[i, 0] = xRes[i][0]
        #     predictTargetXList[i, 1] = yRes[i][0]
        #
        # self.lastTargetX[0], self.lastTargetX[1] = targetX[0], targetX[1]
        # return predictTargetXList


# testTarget = [[i * 10 + 5 * np.random.random(), i * 10 + 5 * np.random.random()] for i  in range(50)]
# tmp = TargetMovePredictor(1)
#
# predictTargetXList = []
#
# tmp.predict(np.array(testTarget[0])).tolist()
# for i in range(1, 2):
#     # predictTargetXList.append(tmp.predict(np.array(testTarget[i])).tolist())
#     # tmp.predict(np.array(testTarget[i])).tolist()
#     tmp.predictMultiSet(testTarget[i], 3)
#
#
# predictTargetXList = tmp.predictMultiSet(testTarget[2], 3)
#
# scattersList = [np.array(testTarget[2:5]), np.array(predictTargetXList)]
# cd = CoorDiagram()
# cd.drwaManyScattersInOnePlane(scattersList)
