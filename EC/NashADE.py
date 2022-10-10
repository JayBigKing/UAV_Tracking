import numpy as np
import time
from copy import deepcopy
from UAV_Common import calcMovingForUAV, calcDistance
import sys

sys.path.append('')
sys.path.append('../logs')
# import myLogger
from DiffEC import DiffEC


class NashADE(DiffEC):
    # def __init__(self, n, dimNum, uDimNum, maxConstraint, minConstraint, fittingLambdaList, F0, F1, CR0, CR1,
    #              timeStepsQ,  dmin, dmax, deltaD, deltaTime,useCuda=False):
    def __init__(self, n, dimNum, uDimNum, maxConstraint, minConstraint, ECArgs):
        # TargetPredictTrajectory, otherAgentX, otherAgentPredictU
        self.init(n, dimNum, uDimNum, maxConstraint, minConstraint, ECArgs["fittingLambdaList"], ECArgs["F0"],
                  ECArgs["F1"], ECArgs["CR0"], ECArgs["CR1"], ECArgs["timeStepsQ"],
                  ECArgs["dmin"], ECArgs["dmax"], ECArgs["deltaD"], ECArgs["deltaTime"], ECArgs["useCuda"])

    def init(self, n, dimNum, uDimNum, maxConstraint, minConstraint, fittingLambdaList, F0, F1, CR0, CR1, timeStepsQ,
             dmin, dmax, deltaD, deltaTime, useCuda=True):
        super().__init__(n, dimNum, maxConstraint, minConstraint, useCuda)
        self.uDimNum = uDimNum
        self.fittingLambdaList = fittingLambdaList
        self.F0 = F0
        self.F1 = F1
        self.CR0 = CR0
        self.CR1 = CR1
        self.timeStepsQ = timeStepsQ
        self.deltaTime = deltaTime
        self.deltaD = deltaD
        self.dMin = dmin
        self.dMax = dmax
        self.__fittingFunctionMap = {0: self.__fittingJTas, 1: self.__fittingJCon, 2: self.__fittingJCol,
                                     3: self.__fittingJCom}
        # print(self.chromosomes)

    def __initChromosomeDim(self):
        if self.useCuda is False:
            for i in range(0, self.Np):
                for j in range(0, self.dimNum):
                    self.chromosomes[j, i] = self.minConstraint[j] + np.random.random() * (
                            self.maxConstraint[j] - self.minConstraint[j])

    def computingPreinit(self, height, agentX):

        self.height = height
        self.agentX = agentX
        self.nowEpochTime = 0
        self.bestChromosomeIndex = 0
        self.firstComputing = True

        self.__initChromosomeDim()

    def computingGetInfoFromOther(self, numOfOtherAgents, otherAgentX, otherAgentPredictU, selfIndex=0):
        self.numOfOtherAgents = numOfOtherAgents
        self.selfIndex = selfIndex
        self.otherAgentX = otherAgentX
        self.otherAgentPredictU = otherAgentPredictU

    def computingGetTargetPredictInfo(self, TargetPredictTrajectory):
        self.TargetPredictTrajectory = TargetPredictTrajectory

    def mutationComputing(self):
        if self.nowEpochTime == 0:
            Fg = self.F0
        elif self.nowEpochTime == 1:
            self.__J_U0 = 0.0
            for i in range(self.Np):
                self.__J_U0 += self.fitting(i)
            Fg = self.F1
        else:
            J_Ug = 0.0
            for i in range(self.Np):
                J_Ug += self.fitting(i)
            Fg = max(self.F1, self.F0 * (J_Ug / self.__J_U0))
        self.mutation(Fg)

    def crossoverComputing(self):
        CRgList = np.zeros(self.Np)
        J_UigList = np.array([self.fitting(i) for i in range(self.Np)])
        J_Uavg = J_UigList.sum() / float(self.Np)
        J_Umin = min(J_UigList)
        for i in range(self.Np):
            if J_UigList[i] > J_Uavg:
                CRgList[i] = self.CR1
            else:
                CRgList[i] = self.CR0 * ((self.CR1 - self.CR0) * (J_Uavg - J_UigList[i])) / (J_Uavg - J_Umin)

        self.crossover(CRgList)

    def cmpFitting(self, val1, val2):
        return (val2 - val1)

    def fitting(self, index):
        fittingValue = 0.0
        for i in range(1):
            fittingValue += self.fittingLambdaList[i] * self.__fittingFunctionMap[i](index)
        return fittingValue

    def computing(self, nowEpoch):
        self.nowEpochTime = nowEpoch
        if self.firstComputing is True:
            self.firstComputing = False
            for i in range(self.Np):
                self.chromosomesVaule[i] = self.fitting(i)
                if self.cmpFitting(self.chromosomesVaule[i], self.chromosomesVaule[self.bestChromosomeIndex]) > 0:
                    self.bestChromosomeIndex = i
        self.mutationComputing()
        self.crossoverComputing()
        self.choose()
        return deepcopy(self.chromosomes[:, self.bestChromosomeIndex]), self.chromosomesVaule[self.bestChromosomeIndex]

    def getBestChromosome(self):
        return deepcopy(self.chromosomes[:, self.bestChromosomeIndex]), self.chromosomesVaule[self.bestChromosomeIndex]

    def __getUOnTimeStep(self, timestep):
        try:
            if timestep < -1 or timestep > self.timeStepsQ:
                raise ValueError('wrong timestep')
        except ValueError as e:
            # myLogger.myLogger_Error(repr(e))
            print(e)
        else:
            # return [i for i in range((timestep) * self.uDimNum, (timestep + 1) * self.uDimNum)]
            return int((timestep) * self.uDimNum), int((timestep + 1) * self.uDimNum)

    def __fittingJTas(self, index):
        LiList = np.zeros(self.timeStepsQ)
        if self.useCuda == False:
            chromosome = self.chromosomes[:, index]
            for i in range(1):
                startIndex, endIndex = self.__getUOnTimeStep(i)
                agentNewX = calcMovingForUAV(self.agentX, chromosome[startIndex: endIndex], self.deltaTime)
                LiList[i] = np.square((agentNewX[0] - self.TargetPredictTrajectory[i][0])) + \
                            np.square(agentNewX[1] - self.TargetPredictTrajectory[i][1]) + np.square(self.height)

        JtasMax = np.max(LiList)
        JtasMin = np.min(LiList)
        allJtas = np.sum(LiList)

        # return (allJtas / self.timeStepsQ - JtasMin) / (JtasMax - JtasMin)
        return allJtas

    def __fittingJCon(self, index):
        JConList = np.zeros((self.timeStepsQ, 2))
        if self.useCuda == False:
            chromosome = self.chromosomes[:, index]
            for i in range(1, self.timeStepsQ):
                startIndex0, endIndex0 = self.__getUOnTimeStep(i - 1)
                startIndex1, endIndex1 = endIndex0, endIndex0 + self.uDimNum
                u0 = chromosome[startIndex0: endIndex0]
                u1 = chromosome[startIndex1: endIndex1]
                JConList[i - 1, 0] = np.abs(u1[0] - u0[0])
                JConList[i - 1, 1] = np.abs(u1[1] - u0[1])
        else:
            pass

        JConMax = np.argmax(JConList, axis=1)
        JConMin = np.argmin(JConList, axis=1)
        allJCon = np.sum(JConList, axis=1)

        return 0.5 * (((allJCon[0] / (self.timeStepsQ - 1) - JConMin[0]) / (JConMax[0] - JConMin[0]))
                      + ((allJCon[1] / (self.timeStepsQ - 1) - JConMin[1]) / (JConMax[1] - JConMin[1]))
                      )

    def __fittingJCol(self, index):
        JCol = 0.0
        chromosome = self.chromosomes[:, index]
        for k in range(self.timeStepsQ):
            for i in range(self.numOfOtherAgents):
                if i != self.selfIndex:
                    startIndex, endIndex = self.__getUOnTimeStep(k)
                    d = calcDistance(calcMovingForUAV(self.agentX, chromosome[startIndex: endIndex], self.deltaTime),
                                     calcMovingForUAV(self.otherAgentX[:, i], self.otherAgentPredictU[:, i],
                                                      self.deltaTime))
                    JCol += 0.5 * (1 + np.tanh(8 * (d - self.dMin - 0.5 * self.deltaD) / self.deltaD))
        return JCol

    def __fittingJCom(self, index):
        JCom = 0.0
        chromosome = self.chromosomes[:, index]
        for k in range(self.timeStepsQ):
            for i in range(self.numOfOtherAgents):
                if i != self.selfIndex:
                    startIndex, endIndex = self.__getUOnTimeStep(k)
                    d = calcDistance(calcMovingForUAV(self.agentX, chromosome[startIndex: endIndex], self.deltaTime),
                                     calcMovingForUAV(self.otherAgentX[:, i],
                                                      self.otherAgentPredictU[startIndex: endIndex, i], self.deltaTime))
                    JCom += 0.5 * (1 + np.tanh(8 * (d - self.dMax - 0.5 * self.deltaD) / self.deltaD))
        return JCom
