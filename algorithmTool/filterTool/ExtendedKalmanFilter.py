#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : ExtendedKalmanFilter.py
@Author  : jay.zhu
@Time    : 2022/10/21 21:42
"""
import math
import numpy as np


class ExtendedKalmanFilter:
    # Covariance for EKF simulation
    __DEFAULT_Q = np.diag([
        .1,  # variance of location on x-axis
        .1,  # variance of location on y-axis
        np.deg2rad(1.0),  # variance of yaw angle
        1.  # variance of velocity
    ]) ** 2  # predict state covariance
    __DEFAULT_R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

    def __init__(self, Q=None, R=None, deltaTime=1.):
        if Q is None:
            self.Q = self.__DEFAULT_Q
        else:
            if isinstance(Q, np.ndarray):
                self.Q = np.array(Q)
            else:
                self.Q = np.diag(Q)

        if R is None:
            self.R = self.__DEFAULT_R
        else:
            if isinstance(R, np.ndarray):
                self.R = R
            else:
                self.R = np.diag(R)

        self.deltaTime = deltaTime
        self.xEst = np.zeros((4, 1))
        self.PEst = np.eye(4)

    @staticmethod
    def motion_model(x, u, deltaTime=1.):
        F = np.array([[x[0, 0]],
                      [x[1, 0]],
                      [x[2, 0]],
                      [0]])

        B = np.array([[deltaTime * math.cos(x[2, 0]) * u[0, 0]],
                      [deltaTime * math.sin(x[2, 0]) * u[0, 0]],
                      [deltaTime * u[1, 0]],
                      [u[0, 0]]])

        return F + B
        # F = np.array([[1.0, 0, 0, 0],
        #               [0, 1.0, 0, 0],
        #               [0, 0, 1.0, 0],
        #               [0, 0, 0, 0]])
        #
        # B = np.array([[deltaTime * math.cos(x[2, 0]), 0],
        #               [deltaTime * math.sin(x[2, 0]), 0],
        #               [0.0, deltaTime],
        #               [1.0, 0.0]])
        #
        # x = F @ x + B @ u
        # return x

    @staticmethod
    def observation_model(x):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = H @ x
        return z

    def jacob_f(self, x, u):
        """
        Jacobian of Motion Model
        motion model
        x_{t+1} = x_t+v*dt*cos(yaw)
        y_{t+1} = y_t+v*dt*sin(yaw)
        yaw_{t+1} = yaw_t+omega*dt
        v_{t+1} = v{t}
        so
        dx/dyaw = -v*dt*sin(yaw)
        dx/dv = dt*cos(yaw)
        dy/dyaw = v*dt*cos(yaw)
        dy/dv = dt*sin(yaw)
        """
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([
            [1.0, 0.0, -self.deltaTime * v * math.sin(yaw), self.deltaTime * math.cos(yaw)],
            [0.0, 1.0, self.deltaTime * v * math.cos(yaw), self.deltaTime * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])

        return jF

    def jacob_h(self):
        if hasattr(self, "jH") is False:
            # Jacobian of Observation Model
            self.jH = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])

        return self.jH

    def predict(self, z, u):
        #  Predict
        xPred = self.motion_model(self.xEst, u, self.deltaTime)
        jF = self.jacob_f(self.xEst, u)
        PPred = jF @ self.PEst @ jF.T + self.Q

        #  Update
        jH = self.jacob_h()
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + self.R
        K = PPred @ jH.T @ np.linalg.inv(S)
        self.xEst = xPred + K @ y
        self.PEst = (np.eye(len(self.xEst)) - K @ jH) @ PPred
        return np.array(self.xEst), np.array(self.PEst)

    def multiPredict(self, z, u, n):
        xEstList = []
        PEstList = []
        xEst, PEst = self.predict(z, u)
        xEstList.append(xEst)
        PEstList.append(PEst)

        originXest = np.array(self.xEst)
        originPest = np.array(self.PEst)
        for i in range(n-1):
            xEst = self.motion_model(xEst, u)
            xEst, PEst = self.predict(xEst[0:2], u)
            xEstList.append(xEst)
            PEstList.append(PEst)
        self.xEst = originXest
        self.PEst = originPest
        return xEstList, PEstList

