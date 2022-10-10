from enum import Enum
from collections import defaultdict

class EC_ChangeDetector_DetectState(Enum):
    FIRST = 0,
    INITIAL = 1,
    ON_WORK = 2,

class EC_ChangeDetector_Base:


    EC_ChangeDetector_DEFAULT_ARGS = {
        "initialNeedTimes": 5
    }

    def __init__(self, n, detectArgs=None):
        self.chromosomesNum = n
        self.firstDetect = True
        self.detectState = self.EC_ChangeDetector_DetectState.FIRST
        self.lastPerformance = 0.
        self.performanceThreshold = 0.
        self.lastChromosomes = None
        self.lastChromosomesFittingVal = None
        self.detectArgs = defaultdict(list)
        if detectArgs is not None:
            self.detectArgs.update(detectArgs)

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

    def analyseInputECData(self, detectState = EC_ChangeDetector_DetectState.FIRST, **kwargs):
        if detectState == EC_ChangeDetector_DetectState.FIRST or detectState == EC_ChangeDetector_DetectState.INITIAL:
            return kwargs["chromosome"], kwargs["chromosomesFittingValue"], kwargs["bestChromosomesFittingValue"]
        elif detectState == EC_ChangeDetector_DetectState.ON_WORK:
            return kwargs["chromosome"], kwargs["chromosomesFittingValue"], kwargs["bestChromosomesFittingValue"]


    def firstDetectProcess(self, kwargs):
        self.lastChromosomes, \
        self.lastChromosomesFittingVal, \
        self.lastPerformance = self.analyseInputECData()

        self.initialRunningTimes = 0
        self.detectState = EC_ChangeDetector_DetectState.INITIAL

    def initialDetectProcess(self, kwargs):
        initialNeedTimes = self.detectArgs["initialNeedTimes"] if self.detectArgs.get("initialNeedTimes") else \
        self.EC_ChangeDetector_DEFAULT_ARGS["initialNeedTimes"]

        if self.initialRunningTimes < initialNeedTimes:
            self.lastChromosomes, \
            self.lastChromosomesFittingVal, \
            self.lastPerformance = self.analyseInputECData()

            self.initialRunningTimes += 1
        else:
            self.detectState = EC_ChangeDetector_DetectState.ON_WORK

    def onWorkDetectProcess(self, kwargs):
        nowChromosome, \
        nowChromosomeFittingVal, \
        bestFittingFuncVal = self.analyseInputECData()

        if bestFittingFuncVal >= self.lastChromosomesFittingVal:
            return False
        else:
            self.detectState = EC_ChangeDetector_DetectState.INITIAL
            return True

class EC_ChangeDetector_EvaluateSolutions(EC_ChangeDetector_Base):
    def __init__(self, n):
        super().__init__(n)


class EC_ChangeDetector_BestSolution(EC_ChangeDetector_Base):
    def __init__(self, n):
        super().__init__(n)
