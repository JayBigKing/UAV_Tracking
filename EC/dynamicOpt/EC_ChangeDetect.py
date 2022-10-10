from enum import Enum
from collections import defaultdict


class EC_ChangeDetector_Base:
    class EC_ChangeDetector_DetectState(Enum):
        FIRST = 0,
        INITIAL = 1,
        ON_WORK = 2,

    EC_ChangeDetector_DEFAULT_ARGS = {
        "initialTimes": 5,
    }

    def __init__(self, n, detect_args=None):
        self.chromosomesNum = n
        self.firstDetect = True
        self.detectState = self.EC_ChangeDetector_DetectState.FIRST
        self.lastPerformance = 0.
        self.performanceThreshold = 0.
        self.lastChromosomes = None
        self.lastChromosomesFittingVal = None
        self.detectArgs = defaultdict(list)
        if detect_args is not None:
            self.detectArgs.update(detect_args)

    @classmethod
    def initByKwargs(cls, **kwargs):
        return cls(kwargs["n"], kwargs)

    def isChange(self, **kwargs):
        if self.detectState == self.EC_ChangeDetector_DetectState.FIRST:
            self.firstDetectProcess(kwargs)
        elif self.detectState == self.EC_ChangeDetector_DetectState.INITIAL:
            self.initialDetectProcess(kwargs)
        elif self.detectState == self.EC_ChangeDetector_DetectState.ON_WORK:
            self.onWorkDetectProcess(kwargs)
        else:
            raise ValueError("detectState just contains FIRST, INITIAL, ON_WORK, which are the value of enum "
                             "EC_ChangeDetector_DetectState")

    def firstDetectProcess(self, **kwargs):
        self.detectState = self.EC_ChangeDetector_DetectState.INITIAL
        nowChromosome = kwargs["chromosome"]
        nowChromosomeFittingVal = kwargs["chromosomesFittingValue"]
        bestFittingFuncVal = kwargs["bestChromosomesFittingValue"]
        self.lastChromosomes = nowChromosome
        self.lastChromosomesFittingVal = nowChromosomeFittingVal
        self.lastPerformance = bestFittingFuncVal

    def initialDetectProcess(self, **kwargs):
        pass

    def onWorkDetectProcess(self, *args):
        pass


class EC_ChangeDetector_EvaluateSolutions(EC_ChangeDetector_Base):
    def __init__(self, n):
        super().__init__(n)


class EC_ChangeDetector_BestSolution(EC_ChangeDetector_Base):
    def __init__(self, n):
        super().__init__(n)
