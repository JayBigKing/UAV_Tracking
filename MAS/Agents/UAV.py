import numpy as np
from EC.NashADE import NashADE
from copy import deepcopy
from Agent_Common import calcMovingForUAV

class My_UAV:
    def __init__(self, initX, height, n, timeStepsQ, maxConstraint, minConstraint, fittingLambdaList, F0, F1, CR0, CR1, dmin,
                 dmax, deltaD, deltaTime=1):
        initX.append(0.)
        self.X = np.array(initX)
        self.u = np.zeros(2)
        self.h = height
        self.timeStepsQ = timeStepsQ
        self.dimNum = 2 * timeStepsQ
        self.predictUList = np.zeros((2, self.timeStepsQ))
        self.EC_PredictArray = np.zeros(self.dimNum)
        self.deltaTime = deltaTime
        self.MaxConstraint = []
        self.MinConstraint = []
        self.fittingLambdaList = fittingLambdaList
        self.F = [F0, F1]
        self.CR = [CR0, CR1]

        for i in range(timeStepsQ):
            self.MaxConstraint.extend(maxConstraint)
            self.MinConstraint.extend(minConstraint)
        self.UAVTrack = NashADE(n, self.dimNum, 2, self.MaxConstraint, self.MinConstraint, fittingLambdaList, F0,
                                      F1, CR0, CR1, timeStepsQ, dmin, dmax, deltaD, deltaTime)

    def updateU(self, u):
        self.u = u

    def moving(self):
        # self.movingMiddleMatrix[0:2, 0] = np.array([[np.cos(self.X[2, 0]) * self.deltaTime],
        #                                             [np.sin(self.X[2, 0]) * self.deltaTime]])
        self.X = calcMovingForUAV(self.X, self.u, self.deltaTime)

    def movingUsingPredictList(self, timeStep):
        self.updateU(self.predictUList[:, timeStep])
        self.moving()

    def getInfoFromOthers(self, numOfOtherAgents, otherAgentX, otherAgentPredictU, selfIndex = 0):
        self.numOfOtherAgents = numOfOtherAgents
        self.otherAgentX = otherAgentX
        self.otherAgentPredictU = otherAgentPredictU
        self.selfIndex = selfIndex
        self.UAVTrack.computingGetInfoFromOther(numOfOtherAgents, otherAgentX, otherAgentPredictU, selfIndex)

    def predictTarget(self, *args):
        self.TargetPredictTrajectory = args[0]
        self.UAVTrack.computingGetTargetPredictInfo(self.TargetPredictTrajectory)

    def predictPreInit(self):
        self.EC_PredictArray = self.UAVTrack.computingPreinit(self.h, self.X)
        self.EC_PredictArray, _ = self.UAVTrack.getBestChromosome()
        self.EC_PredictArrayTransferToPredictUList(self.EC_PredictArray)
        return deepcopy(self.predictUList)


    def computing(self, nowEpoch):
        self.EC_PredictArray, self.predictUFittingValue = self.UAVTrack.computing(nowEpoch)
        self.EC_PredictArrayTransferToPredictUList(self.EC_PredictArray)
        return deepcopy(self.predictUList), self.predictUFittingValue

    def EC_PredictArrayTransferToPredictUList(self, EC_PredictArray):
        for i in range(self.timeStepsQ):
            self.predictUList[0, i] = EC_PredictArray[i * 2]
            self.predictUList[1, i] = EC_PredictArray[i * 2 + 1]

    def __repr__(self):
        return str.format("X:{0:}\nu:{1:}\nh:{2:}", self.X, self.u, self.h)

# uav = My_UAV()
# uav.u[0] = 2
# uav.u[1] = np.radians(45)
# print(repr(uav))
# uav.moving()
# print(repr(uav))
# uav.moving()
# print(repr(uav))
