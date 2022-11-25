#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : UAV_MultiTarget_PredictMAS.py
@Author  : jay.zhu
@Time    : 2022/11/2 17:06
"""
import numpy as np
from algorithmTool.filterTool.ExtendedKalmanFilter import ExtendedKalmanFilter
from EC.EC_Common import ArgsDictValueController
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_MAS_Base import UAV_MultiTarget_MAS_Base


class UAV_MultiTarget_PredictMAS(UAV_MultiTarget_MAS_Base):
    __UAV_MULTI_TARGET_PREDICT_MAS_DEFAULT_ARGS = {
        "kalman_Q": (np.diag([
            .1,  # variance of location on x-axis
            .1,  # variance of location on y-axis
            np.deg2rad(1.0),  # variance of yaw angle
            1.  # variance of velocity
        ]) ** 2),  # predict state covariance,
        "kalman_R": (np.diag([1.0, 1.0]) ** 2)
    }
    def __init__(self, agents, masArgs, targetNum, terminalHandler=None, predictorCls=None, deltaTime=1.):
        super().__init__(agents=agents,
                         masArgs=masArgs,
                         targetNum=targetNum,
                         terminalHandler=terminalHandler,
                         deltaTime=deltaTime)
        self.predictMas_Args = ArgsDictValueController(masArgs, self.__UAV_MULTI_TARGET_PREDICT_MAS_DEFAULT_ARGS)

        if predictorCls is None:
            self.predictorCls = ExtendedKalmanFilter
            self.trajectoryPredictorList = [self.predictorCls(self.predictMas_Args["kalman_Q"],
                                                            self.predictMas_Args["kalman_R"],
                                                            deltaTime)
                                            for i in range(targetNum)]
        else:
            self.predictorCls = predictorCls
            self.trajectoryPredictorList = [self.predictorCls(predictArgs=masArgs) for i in range(targetNum)]

        self.targetPositionList = [np.zeros(2 * self.predictMas_Args["predictVelocityLen"]) for i in range(targetNum)]
        self.uavMovingTime = 0
        self.waitingInitPredictorCount = 0
        self.firstPredict = True

    def recvFromEnv(self, **kwargs):
        # self.targetPosition = kwargs["targetPosition"]
        # self.targerVelocity = kwargs["targetVelocity"]
        # self.trajectoryPredictor.predict(self.targetPosition[0:2], self.targerVelocity)
        super().recvFromEnv(targetPosition=kwargs["targetPosition"], targetVelocity=kwargs["targetVelocity"])
        if self.firstPredict is True:
            for index, trajectoryPredictor in enumerate(self.trajectoryPredictorList):
                trajectoryPredictor.xEst = np.array([[kwargs["targetPosition"][index][0]],
                                                     [kwargs["targetPosition"][index][1]],
                                                     [kwargs["targetPosition"][index][2]],
                                                     [kwargs["targetVelocity"][index][0]], ])
            self.firstPredict = False
        if self.waitingInitPredictorCount < self.predictMas_Args["waitingInitPredictorTime"]:
            for index,trajectoryPredictor in enumerate(self.trajectoryPredictorList):
                trajectoryPredictor.predict(np.vstack(kwargs["targetPosition"][index][0:2]),
                                            np.vstack(kwargs["targetVelocity"][[index]]))
        else:
            for index, trajectoryPredictor in enumerate(self.trajectoryPredictorList):
                if self.agents[0].remainMoving == 0:
                    self.uavMovingTime = self.predictMas_Args["usePredictVelocityLen"]
                    xEstList, _ = trajectoryPredictor.multiPredict(np.vstack(kwargs["targetPosition"][index][0:2]),
                                                                   np.vstack(kwargs["targetVelocity"][index]),
                                                                   self.predictMas_Args["predictVelocityLen"])
                    self.targetPositionList[index] = np.array(
                        [xEstList[i][0:2, :] for i in range(self.predictMas_Args["predictVelocityLen"])])
                else:
                    trajectoryPredictor.predict(np.vstack(kwargs["targetPosition"][index][0:2]),
                                                np.vstack(kwargs["targetVelocity"][index]))

    def update(self):
        if self.waitingInitPredictorCount < self.predictMas_Args["waitingInitPredictorTime"]:
            self.waitingInitPredictorCount += 1
        else:
            super().update()