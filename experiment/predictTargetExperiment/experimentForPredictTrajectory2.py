#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : experimentForPredictTrajectory2.py
@Author  : jay.zhu
@Time    : 2022/10/21 22:19
"""
import math
import random

import numpy as np
from algorithmTool.filterTool.ExtendedKalmanFilter import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation as Rot
from Jay_Tool.visualizeTool.CoorDiagram import CoorDiagram

INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u):
    xTrue = ExtendedKalmanFilter.motion_model(xTrue, u)

    # add noise to gps x-y
    z = ExtendedKalmanFilter.observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = ExtendedKalmanFilter.motion_model(xd, ud)

    return xTrue, z, xd, ud

def observation_no_noise(xTrue, xd, u):
    xTrue = ExtendedKalmanFilter.motion_model(xTrue, u)

    # add noise to gps x-y
    z = ExtendedKalmanFilter.observation_model(xTrue)

    # add noise to input
    ud = u

    xd = ExtendedKalmanFilter.motion_model(xd, ud)

    return xTrue, z, xd, ud

def getInitVar(DT, SIM_TIME = 20.):
    DT = 0.1

    # State Vector [x y yaw v]'
    xTrue = np.zeros((4, 1))

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = np.zeros((4, 1))
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    ekf = ExtendedKalmanFilter(deltaTime=DT)

    return xTrue, xDR, hxEst, hxTrue, hxDR, hz, ekf

def test1():
    import matplotlib.pyplot as plt

    def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
        Pxy = PEst[0:2, 0:2]
        eigval, eigvec = np.linalg.eig(Pxy)

        if eigval[0] >= eigval[1]:
            bigind = 0
            smallind = 1
        else:
            bigind = 1
            smallind = 0

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        a = math.sqrt(eigval[bigind])
        b = math.sqrt(eigval[smallind])
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
        rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
        fx = rot @ (np.array([x, y]))
        px = np.array(fx[0, :] + xEst[0, 0]).flatten()
        py = np.array(fx[1, :] + xEst[1, 0]).flatten()
        plt.plot(px, py, "--r")
    print(__file__ + " start!!")

    DT = 0.1
    SIM_TIME = 20.
    time = 0.0
    xTrue, xDR, hxEst, hxTrue, hxDR, hz, ekf = getInitVar(DT)

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf.predict(z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if True:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

def calcAndOutputOnePredict(DT, SIM_TIME, xTrue, xDR, ekf, calc_input_handler = calc_input, observation_hanlder = observation):
    time = 0.0
    xEstVec = []
    zVec = []

    while SIM_TIME >= time:
        time += DT
        u = calc_input_handler()

        xTrue, z, xDR, ud = observation_hanlder(xTrue, xDR, u)

        xEst, PEst = ekf.predict(z, ud)

        # store data history
        xEstVec.append(np.hstack(xEst[0:2, :]))
        zVec.append(np.hstack(z))

    scattersList = [np.array(zVec), np.array(xEstVec)]
    cd = CoorDiagram()
    cd.drwaManyScattersInOnePlane(scattersList)

def test2():
    DT = 0.1
    SIM_TIME = 20.
    time = 0.0
    xTrue, xDR, *others, ekf = getInitVar(DT)
    calcAndOutputOnePredict(DT, SIM_TIME, xTrue, xDR, ekf)

def test3():
    def calc_zhi_input():
        v = 1.0  # [m/s]
        yawrate = 0  # [rad/s]
        u = np.array([[v], [yawrate]])
        return u
    DT = 0.1
    SIM_TIME = 20.
    time = 0.0
    xTrue, xDR, *others, ekf = getInitVar(DT)
    xTrue[2, 0] = 20.

    calcAndOutputOnePredict(DT, SIM_TIME, xTrue, xDR, ekf, calc_input_handler = calc_zhi_input, observation_hanlder=observation_no_noise)



def testRand1():
    def calc_rand_input():
        v = 5. * random.random()  # [m/s]
        yawrate = 0.5 * random.random()  # [rad/s]
        u = np.array([[v], [yawrate]])
        return u
    DT = 0.1
    SIM_TIME = 20.
    time = 0.0
    xTrue, xDR, *others, ekf = getInitVar(DT)
    calcAndOutputOnePredict(DT, SIM_TIME, xTrue, xDR, ekf, calc_input_handler=calc_rand_input)
lastV = 0.
def testMultiPredict1():

    def calc_zhi_input():
        v = 1.0  # [m/s]
        yawrate = 0  # [rad/s]
        u = np.array([[v], [yawrate]])
        return u
    def calc_rand_input():
        v = 5. * random.random()  # [m/s]
        yawrate = 1. * random.random()  # [rad/s]
        u = np.array([[v], [yawrate]])
        return u
    def calc_rand_input2():
        global lastV
        v = lastV + random.uniform(-2., 2.)  # [m/s]
        lastV = v
        yawrate = random.uniform(-1., 2.)  # [rad/s]
        u = np.array([[v], [yawrate]])
        return u

    calc_input_handler = calc_rand_input2

    DT = 0.1
    SIM_TIME = 20.
    time = 0.0
    xTrue, xDR, *others, ekf = getInitVar(DT)
    xTrue[2,0] = 20

    FIRST_SAMPLE_COUNT_FOR_KALMAN = 3
    PREDICT_STEPS = 3
    HOW_MANY_TIMES = 20
    nowTargetPointIndex = 0

    for i in range(nowTargetPointIndex, 10):
        u = calc_input_handler()
        xTrue, z, xDR, ud = observation_no_noise(xTrue, xDR, u)
        ekf.predict(z, ud)

    for k in range(HOW_MANY_TIMES):

        for i in range(nowTargetPointIndex, FIRST_SAMPLE_COUNT_FOR_KALMAN):
            u = calc_input_handler()
            xTrue, z, xDR, ud = observation_no_noise(xTrue, xDR, u)
            ekf.predict(z, ud)

        xEstVec = []
        zVec = []
        u = calc_input_handler()
        xTrue, z, xDR, ud = observation_no_noise(xTrue, xDR, u)
        xEstList, PEstList = ekf.multiPredict(z, ud, PREDICT_STEPS)
        zVec.append(np.hstack(z))


        for j in range(PREDICT_STEPS - 1):
            u = calc_input_handler()
            xTrue, z, xDR, ud = observation_no_noise(xTrue, xDR, u)
            zVec.append(np.hstack(z))
            ekf.predict(z, ud)

        for j in range(PREDICT_STEPS):
            xEstVec.append(np.hstack(xEstList[j][0:2, :]))

        scattersList = [np.array(zVec), np.array(xEstVec)]
        cd = CoorDiagram()
        cd.drwaManyScattersInOnePlane(scattersList, nameList=["z", "xEst"])
        print(k)
        nowTargetPointIndex += FIRST_SAMPLE_COUNT_FOR_KALMAN


def main():
    testMultiPredict1()
    # test3()
    # testRand1()

if __name__ == "__main__":
    main()