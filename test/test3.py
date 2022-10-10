from Agents.Agent_Common import calcMovingForUAV, calcDistance, clacDirection
from Jay_Tool.visualizeTool.CoorDiagram import  CoorDiagram
import numpy as np

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