import numpy as np
from copy import deepcopy
from Agent_Common import calcMovingForUAV
import sys

sys.path.append("../visualize")
from CoorDiagram import CoorDiagram


class Target:
    def __init__(self, x, speedRange=[1., 5.], angleRange=[0., 40.], deltaTime=1.):
        x.append(0.)
        self.X = np.array(x)
        self.u = np.zeros(2)
        self.speedRange = deepcopy(speedRange)
        self.angleRange = deepcopy(angleRange)
        if self.speedRange[0] > self.speedRange[1]:
            self.speedRange[0], self.speedRange[1] = self.speedRange[1], self.speedRange[0]
        if self.angleRange[0] > self.angleRange[1]:
            self.angleRange[0], self.angleRange[1] = self.angleRange[1], self.angleRange[0]
        self.deltaTime = deltaTime

    def updateU(self, u):
        self.u[0] = u[0]
        self.u[1] = u[1]

    def moving(self):
        # self.movingMiddleMatrix[0:2, 0] = np.array([[np.cos(self.X[2, 0]) * self.deltaTime],
        #                                             [np.sin(self.X[2, 0]) * self.deltaTime]])
        self.X = calcMovingForUAV(self.X, self.u, self.deltaTime)
        return deepcopy(self.X[:2])

    def randMoving(self):
        self.updateU([np.random.uniform(self.speedRange[0], self.speedRange[1]),
                      np.random.uniform(self.angleRange[0], self.angleRange[1])])
        return self.moving()

    def movingAsFunction(self):
        self.X[0] = self.X[0] + self.deltaTime
        self.X[1] = self.X[1] + 4.5 * np.sin(np.deg2rad(self.X[2] * self.deltaTime))
        self.X[2] = self.X[2] + 90 * self.deltaTime
        return deepcopy(self.X[:2])

# x = [1.2, 3.]
# target = Target(x)
# trajectory = [np.array(x[:2])]
# for i in range(10):
#     target.updateU(np.array([2, np.radians((40) * np.random.random())]))
#     trajectory.append(target.moving())
# cd = CoorDiagram()
# cd.drwaManyScattersInOnePlane([trajectory])

# DEFAULT_TARGET_INIT_ANGLE_RANGE = [-180., 180.]
# DEFAULT_TARGET_ANGLE_RANGE = [-500., 500.]
# DEFAULT_TARGET_V_RANGE = [15., 80., np.random.uniform(DEFAULT_TARGET_INIT_ANGLE_RANGE[0], DEFAULT_TARGET_INIT_ANGLE_RANGE[1])]
#
# for j in range(10):
#     x = [1.2, 3.]
#     target = Target(x, DEFAULT_TARGET_V_RANGE, DEFAULT_TARGET_ANGLE_RANGE, 0.05)
#     trajectory = [np.array(x[:2])]
#     for i in range(10):
#         trajectory.append(target.randMoving())
#     cd = CoorDiagram()
#     cd.drwaManyScattersInOnePlane([trajectory])

# for j in range(20):
#     print(random.uniform(1., 10.))

# x = [1.2, 3.]
# target = Target(x, deltaTime=0.05)
# trajectory = [np.array(x[:2])]
# for i in range(100):
#     trajectory.append(target.movingAsFunction())
# cd = CoorDiagram()
# cd.drwaManyScattersInOnePlane([trajectory])