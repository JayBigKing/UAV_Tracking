from EC.EC_WithStat_Base import EC_WithStat_Base


class EC_DynamicOpt_Base(EC_WithStat_Base):
    DEFAULT_EC_DYNAMIC_OPT_ARGS = {
        "populationProportionForEvaluateSolutions": 0.15
    }
    def __init__(self, n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay, needEpochTimes, ECArgs,
                 statRegisters=None, changeDetectorRegisters = "EvaluateSolutions", otherTerminalHandler=None, useCuda=False):
        super().__init__(n, dimNum, maxConstraint, minConstraint, evalVars, otimizeWay,
                         needEpochTimes, ECArgs, statRegisters, otherTerminalHandler, useCuda)
        self.initChangeDetector(changeDetectorRegisters)

    def initChangeDetector(self, changeDetectorRegisters):
        if isinstance(changeDetectorRegisters, str):
            pass

