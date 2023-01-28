#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_PredictMAS.py
@Author  : jay.zhu
@Time    : 2022/10/28 14:15
"""
import numpy as np
from optimization.common.ArgsDictValueController import ArgsDictValueController
from MAS.MultiAgentSystem.UAV_MAS.UAV_MAS_Base import UAV_MAS_Base
from algorithmTool.filterTool.ExtendedKalmanFilter import ExtendedKalmanFilter


class UAV_PredictMAS(UAV_MAS_Base):
    __PREDICT_MAS_DEFAULT_ARGS = {
        "kalman_Q": (np.diag([
            .1,  # variance of location on x-axis
            .1,  # variance of location on y-axis
            np.deg2rad(1.0),  # variance of yaw angle
            1.  # variance of velocity
        ]) ** 2),  # predict state covariance,
        "kalman_R": (np.diag([1.0, 1.0]) ** 2)
    }

    def __init__(self, agents, masArgs, predictorCls=None, deltaTime=1.):
        super().__init__(agents, masArgs, None)
        self.lastAgentOptimizationRes = []
        self.predictMas_Args = ArgsDictValueController(masArgs, self.__PREDICT_MAS_DEFAULT_ARGS)
        self.deltaTime = deltaTime
        if predictorCls is None:
            self.predictorCls = ExtendedKalmanFilter
            self.trajectoryPredictor = self.predictorCls(self.predictMas_Args["kalman_Q"],
                                                         self.predictMas_Args["kalman_R"],
                                                         deltaTime)
        else:
            self.predictorCls = predictorCls
            self.trajectoryPredictor = self.predictorCls(predictArgs=masArgs)

        self.uavMovingTime = 0
        self.waitingInitPredictorCount = 0
        self.firstPredict = True

    def recvFromEnv(self, **kwargs):
        # self.targetPosition = kwargs["targetPosition"]
        # self.targerVelocity = kwargs["targetVelocity"]
        # self.trajectoryPredictor.predict(self.targetPosition[0:2], self.targerVelocity)
        if self.firstPredict is True:
            self.trajectoryPredictor.xEst = np.array([[kwargs["targetPosition"][0]],
                                                      [kwargs["targetPosition"][1]],
                                                      [kwargs["targetPosition"][2]],
                                                      [kwargs["targetVelocity"][0]],])
            self.firstPredict = False
        if self.waitingInitPredictorCount < self.predictMas_Args["waitingInitPredictorTime"]:
            self.trajectoryPredictor.predict(np.vstack(kwargs["targetPosition"][0:2]),
                                             np.vstack(kwargs["targetVelocity"]))
        else:
            # if self.uavMovingTime == 0:
            #     self.uavMovingTime = self.predictMas_Args["usePredictVelocityLen"]
            #     xEstList, _ = self.trajectoryPredictor.multiPredict(np.vstack(kwargs["targetPosition"][0:2]),
            #                                                         np.vstack(kwargs["targetVelocity"]),
            #                                                         self.predictMas_Args["predictVelocityLen"])
            #     self.targetPosition = [xEstList[i][0:2] for i in range(self.predictMas_Args["predictVelocityLen"])]
            # else:
            #     self.trajectoryPredictor.predict(np.vstack(kwargs["targetPosition"][0:2]),
            #                                      np.vstack(kwargs["targetVelocity"]))
            #
            # self.uavMovingTime -= 1

            if self.agents[0].remainMoving == 0:
                self.uavMovingTime = self.predictMas_Args["usePredictVelocityLen"]
                xEstList, _ = self.trajectoryPredictor.multiPredict(np.vstack(kwargs["targetPosition"][0:2]),
                                                                    np.vstack(kwargs["targetVelocity"]),
                                                                    self.predictMas_Args["predictVelocityLen"])
                self.targetPosition = [xEstList[i][0:2, :] for i in range(self.predictMas_Args["predictVelocityLen"])]
            else:
                self.trajectoryPredictor.predict(np.vstack(kwargs["targetPosition"][0:2]),
                                                 np.vstack(kwargs["targetVelocity"]))



    def update(self):
        if self.waitingInitPredictorCount < self.predictMas_Args["waitingInitPredictorTime"]:
            self.waitingInitPredictorCount += 1
        else:
            super().update()
