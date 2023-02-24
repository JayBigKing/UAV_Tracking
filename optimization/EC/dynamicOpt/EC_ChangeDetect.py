#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : EC_ChangeDetect.py
@Author  : jay.zhu
@Time    : 2022/10/10 21:58
"""

from enum import Enum
from collections import defaultdict
import random
import numpy as np
from optimization.common.ArgsDictValueController import ArgsDictValueController


class EC_ChangeDetector_DetectState(Enum):
    FIRST = 0,
    INITIAL = 1,
    ON_WORK = 2,

def EC_ChangeDetector_Base_fittingCmpFunc(a, b):
    return a - b

# "fittingCmpFunc": lambda a, b: (a - b),
class EC_ChangeDetector_Base:
    EC_ChangeDetector_DEFAULT_ARGS = {
        "initialNeedTimes": 5,
        "fittingCmpFunc": EC_ChangeDetector_Base_fittingCmpFunc,
    }

    def __init__(self, n, detectArgs=None):
        self.chromosomesNum = n
        self.firstDetect = True
        self.detectState = EC_ChangeDetector_DetectState.FIRST
        self.lastPerformance = 0.
        self.performanceThreshold = 0.
        self.lastChromosomes = None
        self.lastChromosomesFittingVal = None
        self.lastBestChromosomesFittingValue = 0.
        self.argsDictValueController = ArgsDictValueController(detectArgs, self.EC_ChangeDetector_DEFAULT_ARGS)
        self.detectArgs = self.argsDictValueController.userArgsDict
        self.detectArgs = defaultdict(list)
        if detectArgs is not None:
            self.detectArgs.update(detectArgs)
        self.fittingCmpFunc = self.argsDictValueController["fittingCmpFunc"]

    @classmethod
    def initByKwargs(cls, **kwargs):
        return cls(kwargs["n"], kwargs)

    def isChange(self, **kwargs):
        if self.detectState == EC_ChangeDetector_DetectState.FIRST:
            self.firstDetectProcess(kwargs)
            return False
        elif self.detectState == EC_ChangeDetector_DetectState.INITIAL:
            self.initialDetectProcess(kwargs)
            return False
        elif self.detectState == EC_ChangeDetector_DetectState.ON_WORK:
            return self.onWorkDetectProcess(kwargs)
        else:
            raise ValueError("detectState just contains FIRST, INITIAL, ON_WORK, which are the value of enum "
                             "EC_ChangeDetector_DetectState")

    def analyseInputECData(self, kwargs, detectState=EC_ChangeDetector_DetectState.FIRST):
        if detectState == EC_ChangeDetector_DetectState.FIRST or detectState == EC_ChangeDetector_DetectState.INITIAL:
            return kwargs["chromosome"], kwargs["chromosomesFittingValue"], kwargs["bestChromosomesFittingValue"]
        elif detectState == EC_ChangeDetector_DetectState.ON_WORK:
            return kwargs["chromosome"], kwargs["chromosomesFittingValue"], kwargs["bestChromosomesFittingValue"]

    def firstDetectProcess(self, kwargs):
        self.lastChromosomes, self.lastChromosomesFittingVal, self.lastBestChromosomesFittingValue = self.analyseInputECData(
            kwargs, self.detectState)

        self.initialRunningTimes = 0
        self.detectState = EC_ChangeDetector_DetectState.INITIAL

    def initialDetectProcess(self, kwargs):
        initialNeedTimes = self.argsDictValueController["initialNeedTimes"]

        if self.initialRunningTimes < initialNeedTimes:
            self.lastChromosomes, self.lastChromosomesFittingVal, self.lastBestChromosomesFittingValue = self.analyseInputECData(
                kwargs, self.detectState)

            self.initialRunningTimes += 1
        else:
            self.detectState = EC_ChangeDetector_DetectState.ON_WORK

    def onWorkDetectProcess(self, kwargs):
        nowChromosome, nowChromosomeFittingVal, bestFittingFuncVal = self.analyseInputECData(kwargs, self.detectState)

        if self.fittingCmpFunc(bestFittingFuncVal, self.lastBestChromosomesFittingValue) > 0:
            self.lastChromosomes, \
            self.lastChromosomesFittingVal = nowChromosome, nowChromosomeFittingVal
            self.lastBestChromosomesFittingValue, self.lastPerformance = bestFittingFuncVal
            return False
        else:
            self.detectState = EC_ChangeDetector_DetectState.INITIAL
            return True


class EC_ChangeDetector_EvaluateSolutions(EC_ChangeDetector_Base):
    EC_ChangeDetector_EvaluateSolutions_DEFAULT_ARGS = {
        "populationProportionForEvaluateSolutions": 0.2,
        "degradationTimesThreshold": 4,
    }

    def __init__(self, n, detectArgs=None):
        super().__init__(n, detectArgs)
        self.argsDictValueController.update(self.EC_ChangeDetector_EvaluateSolutions_DEFAULT_ARGS)
        if self.chromosomesNum < 10:
            self.inspectedNum = 1
        elif self.chromosomesNum < 200:
            populationProportion = self.argsDictValueController.getValueByKey(
                "populationProportionForEvaluateSolutions")
            self.inspectedNum = int(float(self.chromosomesNum) * populationProportion)
        else:
            self.inspectedNum = 80

        self.inspectedIndexs = random.sample(range(self.chromosomesNum), self.inspectedNum)
        self.lastAvgFittingOfInspectedChromosomes = 0.

    def firstDetectProcess(self, kwargs):
        super().firstDetectProcess(kwargs)
        self.lastAvgFittingOfInspectedChromosomes = np.average(
            self.lastChromosomesFittingVal.take(self.inspectedIndexs))

    def initialDetectProcess(self, kwargs):
        super().initialDetectProcess(kwargs)
        self.lastAvgFittingOfInspectedChromosomes = np.average(
            self.lastChromosomesFittingVal.take(self.inspectedIndexs))
        self.degradationTimes = 0

    def onWorkDetectProcess(self, kwargs):
        _, nowChromosomeFittingVal, _ = self.analyseInputECData(kwargs, self.detectState)

        nowAvgFittingOfInspectedChromosomes = np.average(self.lastChromosomesFittingVal.take(self.inspectedIndexs))
        if self.fittingCmpFunc(nowAvgFittingOfInspectedChromosomes, self.lastAvgFittingOfInspectedChromosomes) < 0:
            self.degradationTimes += 1
            degradationTimesThreshold = self.argsDictValueController("degradationTimesThreshold")
            if self.degradationTimes < degradationTimesThreshold:
                self.lastAvgFittingOfInspectedChromosomes = nowAvgFittingOfInspectedChromosomes
                return False
            else:
                self.detectState = EC_ChangeDetector_DetectState.INITIAL
                return True
        else:
            self.lastAvgFittingOfInspectedChromosomes = nowAvgFittingOfInspectedChromosomes
            self.degradationTimes = 0
            return False


class EC_ChangeDetector_BestSolution(EC_ChangeDetector_Base):
    EC_ChangeDetector_BestSolution_DEFAULT_ARGS = {
        "statisticPeriod": 5,
        "degradationValueThreshold": 1,
    }

    def __init__(self, n, detectArgs=None):
        super().__init__(n, detectArgs)
        self.argsDictValueController.update(self.EC_ChangeDetector_BestSolution_DEFAULT_ARGS)
        self.statisticPeriod = self.argsDictValueController.getValueByKey("statisticPeriod")
        self.argsDictValueController.defaultArgsDict["initialNeedTimes"] = self.statisticPeriod
        self.bestSolutionFittingsInPeriod = np.zeros(self.statisticPeriod)
        self.bestSolutionFittingsPoint = 0

    def firstDetectProcess(self, kwargs):
        super().firstDetectProcess(kwargs)

        self.nowGenOfPeriod = 0
        self.bestSolutionFittingsPoint = 0

    def initialDetectProcess(self, kwargs):
        super().initialDetectProcess(kwargs)
        self.bestSolutionFittingsInPeriod[self.bestSolutionFittingsPoint] = self.lastBestChromosomesFittingValue
        self.bestSolutionFittingsPoint = (self.bestSolutionFittingsPoint + 1) % self.statisticPeriod
        if self.detectState == EC_ChangeDetector_DetectState.ON_WORK:
            self.lastPerformance = np.average(self.bestSolutionFittingsInPeriod)

    def onWorkDetectProcess(self, kwargs):
        _, _, bestFittingFuncVal = self.analyseInputECData(kwargs, self.detectState)
        self.bestSolutionFittingsInPeriod[self.bestSolutionFittingsPoint] = bestFittingFuncVal
        self.bestSolutionFittingsPoint = (self.bestSolutionFittingsPoint + 1) % self.statisticPeriod

        self.nowGenOfPeriod += 1
        if self.nowGenOfPeriod < self.statisticPeriod:
            return False
        else:
            self.nowGenOfPeriod = 0
            nowPerformance = np.average(self.bestSolutionFittingsInPeriod)
            if self.fittingCmpFunc(nowPerformance, self.lastPerformance) > 0 :
                self.lastPerformance = nowPerformance
                return False
            else:
                self.detectState = EC_ChangeDetector_DetectState.INITIAL
                return True


class EC_ChangeDetector_PerformanceThresh(EC_ChangeDetector_Base):
    EC_CHANGEDETECTOR_PERFORMANCETHRESH_DEFAULT_ARGS = {
        "performanceThreshold": 5
    }

    def __init__(self, n, detectArgs=None):
        super().__init__(n, detectArgs)
        if detectArgs.get("performanceThreshold"):
            self.argsDictValueController.update(self.EC_CHANGEDETECTOR_PERFORMANCETHRESH_DEFAULT_ARGS)
        else:
            raise ValueError("detectArgs need a key-value pair which key named performanceThreshold")

    def onWorkDetectProcess(self, kwargs):
        _, _, self.lastPerformance = self.analyseInputECData(kwargs, self.detectState)

        if self.fittingCmpFunc(self.lastPerformance, self.argsDictValueController["performanceThreshold"]) > 0 :
            return False
        else:
            return True
