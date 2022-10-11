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
from EC.EC_Common import ArgsDictValueGetter

class EC_ChangeDetector_DetectState(Enum):
    FIRST = 0,
    INITIAL = 1,
    ON_WORK = 2,

class EC_ChangeDetector_Base:

    EC_ChangeDetector_DEFAULT_ARGS = {
        "initialNeedTimes": 5,
        "fittingCmpFunc": lambda a,b : (a - b),
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
        self.argsDictValueGetter = ArgsDictValueGetter(detectArgs, self.EC_ChangeDetector_DEFAULT_ARGS)
        self.detectArgs = self.argsDictValueGetter.userArgsDict
        self.detectArgs = defaultdict(list)
        if detectArgs is not None:
            self.detectArgs.update(detectArgs)
        self.fittingCmpFunc = self.argsDictValueGetter.getValueByKey("fittingCmpFunc")

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
            self.onWorkDetectProcess(kwargs)
        else:
            raise ValueError("detectState just contains FIRST, INITIAL, ON_WORK, which are the value of enum "
                             "EC_ChangeDetector_DetectState")

    def analyseInputECData(self, kwargs, detectState = EC_ChangeDetector_DetectState.FIRST):
        if detectState == EC_ChangeDetector_DetectState.FIRST or detectState == EC_ChangeDetector_DetectState.INITIAL:
            return kwargs["chromosome"], kwargs["chromosomesFittingValue"], kwargs["bestChromosomesFittingValue"]
        elif detectState == EC_ChangeDetector_DetectState.ON_WORK:
            return kwargs["chromosome"], kwargs["chromosomesFittingValue"], kwargs["bestChromosomesFittingValue"]


    def firstDetectProcess(self, kwargs):
        self.lastChromosomes, \
        self.lastChromosomesFittingVal, \
        self.lastBestChromosomesFittingValue = self.analyseInputECData(self.detectState, kwargs)

        self.initialRunningTimes = 0
        self.detectState = EC_ChangeDetector_DetectState.INITIAL

    def initialDetectProcess(self, kwargs):
        initialNeedTimes = self.argsDictValueGetter.getValueByKey("initialNeedTimes")

        if self.initialRunningTimes < initialNeedTimes:
            self.lastChromosomes, \
            self.lastChromosomesFittingVal, \
            self.lastBestChromosomesFittingValue = self.analyseInputECData(self.detectState, kwargs)

            self.initialRunningTimes += 1
        else:
            self.detectState = EC_ChangeDetector_DetectState.ON_WORK

    def onWorkDetectProcess(self, kwargs):
        nowChromosome, \
        nowChromosomeFittingVal, \
        bestFittingFuncVal = self.analyseInputECData(self.detectState, kwargs)

        if self.fittingCmpFunc(bestFittingFuncVal, self.lastBestChromosomesFittingValue):
            self.lastChromosomes, \
            self.lastChromosomesFittingVal= nowChromosome, nowChromosomeFittingVal
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
        self.argsDictValueGetter.update(defaultArgsDict=self.EC_ChangeDetector_EvaluateSolutions_DEFAULT_ARGS)
        if self.inspectedNum < 10:
            self.inspectedNum = 1
        elif self.inspectedNum < 200:
            populationProportion = self.argsDictValueGetter.getValueByKey("populationProportionForEvaluateSolutions")
            self.inspectedNum = int(float(self.chromosomesNum) * populationProportion)
        else:
            self.inspectedNum = 80

        self.inspectedIndexs = random.sample(range(self.chromosomesNum), self.inspectedNum)
        self.lastAvgFittingOfInspectedChromosomes = 0.

    def firstDetectProcess(self, kwargs):
        super().firstDetectProcess(kwargs)
        self.lastAvgFittingOfInspectedChromosomes = np.average(self.lastChromosomesFittingVal.take(self.inspectedIndexs))

    def initialDetectProcess(self, kwargs):
        super().initialDetectProcess(kwargs)
        self.lastAvgFittingOfInspectedChromosomes = np.average(
            self.lastChromosomesFittingVal.take(self.inspectedIndexs))
        self.degradationTimes = 0

    def onWorkDetectProcess(self, kwargs):
        _, \
        nowChromosomeFittingVal, \
        _ = self.analyseInputECData(self.detectState, kwargs)

        nowAvgFittingOfInspectedChromosomes = np.average(self.lastChromosomesFittingVal.take(self.inspectedIndexs))
        if self.fittingCmpFunc(nowAvgFittingOfInspectedChromosomes, self.lastAvgFittingOfInspectedChromosomes) < 0:
            self.degradationTimes += 1
            degradationTimesThreshold = self.argsDictValueGetter("degradationTimesThreshold")
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
        self.argsDictValueGetter.update(defaultArgsDict=self.EC_ChangeDetector_BestSolution_DEFAULT_ARGS)
        self.statisticPeriod = self.argsDictValueGetter.getValueByKey("statisticPeriod")
        self.argsDictValueGetter.defaultArgsDict["initialNeedTimes"] = self.statisticPeriod
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
        _, \
        _, \
        bestFittingFuncVal = self.analyseInputECData(self.detectState, kwargs)
        self.bestSolutionFittingsInPeriod[self.bestSolutionFittingsPoint] = bestFittingFuncVal
        self.bestSolutionFittingsPoint = (self.bestSolutionFittingsPoint + 1) % self.statisticPeriod

        self.nowGenOfPeriod += 1
        if self.nowGenOfPeriod < self.statisticPeriod:
            return False
        else:
            self.nowGenOfPeriod = 0
            nowPerformance = np.average(self.bestSolutionFittingsInPeriod)
            if self.fittingCmpFunc(nowPerformance, self.lastPerformance):
                self.lastPerformance = nowPerformance
                return False
            else:
                self.detectState = EC_ChangeDetector_DetectState.INITIAL
                return True

