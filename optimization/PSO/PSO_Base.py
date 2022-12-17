#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : PSO_Base.py
@Author  : jay.zhu
@Time    : 2022/12/15 16:05
"""
from enum import Enum
import random
import numpy as np
from optimization.common.optimizationCommonEnum import OptimizationWay
from optimization.common.ArgsDictValueController import ArgsDictValueController
import optimization.common.optimizationCommonFunctions as ocf


class PSO_Base:
    PERSON_NOW_FITNESS_INDEX = 0
    PERSON_BEST_FITNESS_INDEX = 1

    BEST_IN_ALL_GEN_DIM_INDEX = 0
    BEST_IN_NOW_GEN_DIM_INDEX = 1

    __DEFAULT_PSO_BASE_ARGS = {
        "fittingMinDenominator": 0.2,
        "w": 0.6,
        "c1": 0.6,
        "c2": 0.3,
        "velocityFactor": 0.1
    }

    def __init__(self, n, dimNum, positionMaxConstraint, positionMinConstraint, evalVars, optimizeWay, needEpochTimes,
                 PSOArgs, velocityMaxConstraint=None, velocityMinConstraint=None, otherTerminalHandler=None,
                 useCuda=False):
        self.Np = n
        self.dimNum = dimNum
        self.positionMaxConstraint = positionMaxConstraint
        self.positionMinConstraint = positionMinConstraint
        self.PSOArgsDictValueController = ArgsDictValueController(PSOArgs, self.__DEFAULT_PSO_BASE_ARGS)
        if velocityMaxConstraint is None:
            self.velocityMaxConstraint = [item * self.PSOArgsDictValueController["velocityFactor"] for item in self.positionMaxConstraint]
        else:
            self.velocityMaxConstraint = velocityMaxConstraint

        if velocityMinConstraint is None:
            self.velocityMinConstraint = [item * self.PSOArgsDictValueController["velocityFactor"] for item in self.positionMinConstraint]
        else:
            self.velocityMinConstraint = velocityMinConstraint

        self.evalVars = evalVars
        if isinstance(optimizeWay, OptimizationWay) is False:
            raise TypeError("otimizeWay should be an instance of enum OptimizationWay")
        else:
            self.optimizeWay = optimizeWay

        self.particlePositions = np.zeros((dimNum, n))
        self.particleVelocities = np.zeros((dimNum, n))

        self.personBestParticlePositions = np.zeros((dimNum, n))
        self.globalBestParticlePosition = np.zeros((dimNum, 2))

        self.particlesFittingValue = np.zeros((n, 2))  # 包含当前的fitness和个体的历史最优fitness
        self.particlesAimFuncValue = np.zeros((n, 2))  # 包含当前的AimFunc和个体的历史最优AimFunc
        self.globalBestFittingValue = np.zeros(2)
        self.globalBestAimFuncValue = np.zeros(2)

        self._nowEpochTime = 0
        self.needEpochTimes = needEpochTimes

        self.PSOArgs = self.PSOArgsDictValueController.userArgsDict
        self.borders = self.PSOArgs["borders"]  # 如果没有self.ECArgs["borders"] 的话，self.borders就是[]
        self.otherTerminalHandler = otherTerminalHandler
        self.useCuda = useCuda

        # self.particlesInit()
        self.firstRun = True

    def particlesInit(self):
        for i in range(0, self.Np):
            for j in range(0, self.dimNum):
                self.particlePositions[j, i] = self.positionMinConstraint[
                                                   j] + np.random.random() * (
                                                       self.positionMaxConstraint[j] -
                                                       self.positionMinConstraint[j])

                self.particleVelocities[j, i] = self.velocityMinConstraint[
                                                    j] + np.random.random() * (
                                                        self.velocityMaxConstraint[j] -
                                                        self.velocityMinConstraint[j])

    def initShouldContinueVar(self, otherTerminalHandler=None):
        if otherTerminalHandler is not None:
            otherTerminalHandler(initFlag=True)
        self._nowEpochTime = 0

    def optimization(self):
        if self.firstRun == True:
            self.particlesInit()
            self.fitting()
            self.firstRun = False
        self.initShouldContinueVar(self.otherTerminalHandler)
        while self.shouldContinue(self.otherTerminalHandler):
            self.optimizeInner()

        return np.array(self.globalBestParticlePosition[:, self.BEST_IN_ALL_GEN_DIM_INDEX]), \
               self.globalBestAimFuncValue[self.BEST_IN_ALL_GEN_DIM_INDEX]

    def optimizeInner(self):
        self.clearBestChromosome(self.BEST_IN_NOW_GEN_DIM_INDEX)
        self.update()
        self.fitting()

    def update(self):
        w = self.PSOArgsDictValueController["w"]
        c1 = self.PSOArgsDictValueController["c1"]
        c2 = self.PSOArgsDictValueController["c2"]
        for i in range(self.Np):
            for j in range(self.dimNum):
                self.particleVelocities[j, i] = w * self.particleVelocities[j, i] + \
                                                c1 * random.random() * (self.personBestParticlePositions[j, i] -
                                                                        self.particlePositions[j, i]) + \
                                                c2 * random.random() * (self.globalBestParticlePosition[j, self.BEST_IN_ALL_GEN_DIM_INDEX] -self.particlePositions[j, i])
                self.particlePositions[j, i] = self.limitParticleValue(self.particlePositions[j, i] + self.particleVelocities[j, i], j)

    def fitting(self):
        for i in range(0, self.Np):
            self.particlesFittingValue[i, self.PERSON_NOW_FITNESS_INDEX], self.particlesAimFuncValue[
                i, self.PERSON_NOW_FITNESS_INDEX] = ocf.fittingOne(
                solution=self.particlePositions[:, i],
                evalVars=self.evalVars,
                optimizeWay=self.optimizeWay,
                fittingMinDenominator=
                self.PSOArgsDictValueController[
                    "fittingMinDenominator"])

            self.cmpToBestChromosomeAndStore(i)

    def cmpToBestChromosomeAndStore(self, particleIndex):
        if self.cmpFitting(self.particlesFittingValue[particleIndex, self.PERSON_NOW_FITNESS_INDEX],
                           self.particlesFittingValue[particleIndex, self.PERSON_BEST_FITNESS_INDEX]) > 0:
            self.particlesFittingValue[particleIndex, self.PERSON_BEST_FITNESS_INDEX] = self.particlesFittingValue[
                particleIndex, self.PERSON_NOW_FITNESS_INDEX]
            self.particlesAimFuncValue[particleIndex, self.PERSON_BEST_FITNESS_INDEX] = self.particlesAimFuncValue[
                particleIndex, self.PERSON_NOW_FITNESS_INDEX]
            self.personBestParticlePositions[:, particleIndex] = np.array(
                self.particlePositions[:, particleIndex])


            if self.cmpFitting(self.particlesFittingValue[particleIndex, self.PERSON_BEST_FITNESS_INDEX],
                               self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX]) > 0:
                self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX] = self.particlesFittingValue[particleIndex, self.PERSON_BEST_FITNESS_INDEX]
                self.globalBestAimFuncValue[self.BEST_IN_NOW_GEN_DIM_INDEX] = self.particlesAimFuncValue[particleIndex, self.PERSON_BEST_FITNESS_INDEX]
                self.globalBestParticlePosition[:, self.BEST_IN_NOW_GEN_DIM_INDEX] = np.array(
                    self.particlePositions[:, particleIndex])

                if self.cmpFitting(self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX],
                                   self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX]) > 0:
                    self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX],self.globalBestAimFuncValue[self.BEST_IN_ALL_GEN_DIM_INDEX]  = self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX],self.globalBestAimFuncValue[self.BEST_IN_NOW_GEN_DIM_INDEX]
                    self.globalBestParticlePosition[:, self.BEST_IN_ALL_GEN_DIM_INDEX] = np.array(self.globalBestParticlePosition[:, self.BEST_IN_NOW_GEN_DIM_INDEX])


    def cmpFitting(self, val1, val2):
        return (val1 - val2)

    def shouldContinue(self, otherTerminalHandler):
        shouldContinueFlag, self._nowEpochTime = ocf.shouldContinue(nowEpochTime=self._nowEpochTime,
                                                                    bestFittingValue=self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX],
                                                                    bestAimFuncValue=self.globalBestAimFuncValue[self.BEST_IN_ALL_GEN_DIM_INDEX],
                                                                    needEpochTimes=self.needEpochTimes,
                                                                    otherTerminalHandler=otherTerminalHandler)
        return shouldContinueFlag

    def clearBestChromosome(self, whichBestParticle):
        if whichBestParticle == self.BEST_IN_NOW_GEN_DIM_INDEX:
            self.globalBestFittingValue[self.BEST_IN_NOW_GEN_DIM_INDEX], self.globalBestAimFuncValue[
                self.BEST_IN_NOW_GEN_DIM_INDEX] = 0., 0.
        elif whichBestParticle == self.BEST_IN_ALL_GEN_DIM_INDEX:
            self.globalBestFittingValue[self.BEST_IN_ALL_GEN_DIM_INDEX], self.globalBestAimFuncValue[
                self.BEST_IN_ALL_GEN_DIM_INDEX] = 0., 0.

    def limitParticleValue(self, particleValue, dimIndex):
        if self.borders == [] or self.borders[dimIndex] != 1:
            return particleValue
        else:
            return ocf.limitValue(particleValue, dimIndex, self.positionMaxConstraint, self.positionMinConstraint)
