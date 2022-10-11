from EC.EC_WithStat_Base import EC_WithStat_Base
from EC.dynamicOpt.EC_ChangeDetect import EC_ChangeDetector_Base, EC_ChangeDetector_EvaluateSolutions, \
    EC_ChangeDetector_BestSolution


class EC_DynamicOpt_Base(EC_WithStat_Base):
    EC_DYNAMIC_OPT_BASE_DEFAULT_ARGS = {
        "populationProportionForEvaluateSolutions": 0.15,
    }
    EC_DYNAMIC_OPT_BASE_DEFAULT_CHANGE_DETECTOR_REG_DICT = {
        "EvaluateSolutions": EC_ChangeDetector_EvaluateSolutions,
        "AvgBestSolution": EC_ChangeDetector_BestSolution
    }

    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, changeDetectorRegisters="EvaluateSolutions", otherTerminalHandler=None,
                 useCuda=False):
        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay,
                         needEpochTimes, ECArgs, statRegisters, otherTerminalHandler, useCuda)
        self.ECArgsDictValueGetter.update(defaultArgsDict=self.EC_DYNAMIC_OPT_BASE_DEFAULT_ARGS)
        self.initChangeDetector(changeDetectorRegisters)


    def initChangeDetector(self, changeDetectorRegisters):
        if isinstance(changeDetectorRegisters, str):
            if self.EC_DYNAMIC_OPT_BASE_DEFAULT_CHANGE_DETECTOR_REG_DICT.get(changeDetectorRegisters):
                self.changeDetector = self.EC_DYNAMIC_OPT_BASE_DEFAULT_CHANGE_DETECTOR_REG_DICT[changeDetectorRegisters]
            else:
                raise ValueError("no such default change detector"
                                 "maybe you can use a change detector created by yourself")
        else:
            self.changeDetector = changeDetectorRegisters

    def optimizeInner(self):
        super().optimizeInner()
        self.adaptToEnvironment(self.changeDetector.isChange(chromosome=self.chromosomes,
                                                             chromosomesFittingValue=self.chromosomesFittingValue[:,
                                                                                     self.CHROMOSOME_DIM_INDEX],
                                                             bestChromosomesFittingValue=self.chromosomesFittingValue[
                                                                                        -1, self.CHROMOSOME_DIM_INDEX]))

    def adaptToEnvironment(self, isChange):
        pass
