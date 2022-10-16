from Jay_Tool.visualizeTool.CoorDiagram import  CoorDiagram
import numpy as np
def calcMovingForUAV(x, u, time):
    res = np.zeros(3)
    res[0] = x[0] + u[0] * np.cos(np.deg2rad(x[2])) * time
    res[1] = x[1] + u[0] * np.sin(np.deg2rad(x[2])) * time
    res[2] = x[2] + u[1] * time
    return res

def calcDistance(x0, x1):
    return np.sqrt(np.sum(np.square(x0 - x1)))

def clacDirection(x0, x1):
    return np.rad2deg(np.arctan2(x0[1] - x1[1], x0[0] - x1[0]))

def test1():
    target = np.array([np.random.uniform(30, 200), np.random.uniform(30, 200)])
    uav = np.array([np.random.uniform(1, 10), np.random.uniform(1, 10), np.random.uniform(-180, 180)])
    uav_v = np.array([2, 0])
    targetScatters = [[target[0], target[1]]]
    uavScatters = [uav[0:2]]
    while True:
        print(calcDistance(target, uav[0:2]))
        if calcDistance(target, uav[0:2]) < 2:
            break
        else:
            direction = clacDirection(target, uav[0:2])
            if direction - uav[2] > 0:
                uavDirectionSpeed = direction - uav[2] if direction - uav[2] <= 30. else 30.
            else:
                uavDirectionSpeed = direction - uav[2] if uav[2] - direction <= 30. else -30.

            uav_v[1] = uavDirectionSpeed
            uav = calcMovingForUAV(uav, uav_v, 1)
            uavScatters.append(uav[0:2])
            # print(uavDirectionSpeed)

    cd = CoorDiagram()
    scattersList = [targetScatters, uavScatters]
    cd.drwaManyScattersInOnePlane(scattersList)

test1()