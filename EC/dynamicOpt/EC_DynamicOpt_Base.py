#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : uav tracking
@File    : EC_DynamicOpt_Base.py
@Author  : jay.zhu
@Time    : 2022/10/10 20:58
"""

from EC.EC_WithStat_Base import EC_WithStat_Base
from EC.dynamicOpt.EC_ChangeDetect import EC_ChangeDetector_EvaluateSolutions, \
    EC_ChangeDetector_BestSolution


class EC_DynamicOpt_Base(EC_WithStat_Base):
    EC_DYNAMIC_OPT_BASE_DEFAULT_ARGS = {
        "populationProportionForEvaluateSolutions": 0.15,
    }
    EC_DYNAMIC_OPT_BASE_DEFAULT_CHANGE_DETECTOR_REG_DICT = {
        "EvaluateSolutions": EC_ChangeDetector_EvaluateSolutions,
        "AvgBestSolution": EC_ChangeDetector_BestSolution,
        "refractoryPeriodLength":10,
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
    def adaptToEnvironmentWhenChange(self):
        self.refractoryPeriodTick = 0

    def adaptToEnvironmentWhenRefractoryPeriod(self):
        '''
        当遇到环境变化后，EC算法会通过一系列的动作调整种群与算法
        这个时候可能会导致系统fitting出现一些震荡，进而导致change由被识别出来
        所以设置这个不应期RefractoryPeriod，无视检测器监测出来的变化
        '''
        pass

    def adaptToEnvironment(self, isChange):
        if isChange is True:
            self.adaptToEnvironmentWhenChange()
        else:
            refractoryPeriodLength = self.ECArgsDictValueGetter("refractoryPeriodLength")
            if self.refractoryPeriodTick < refractoryPeriodLength:
                self.refractoryPeriodTick += 1
                self.adaptToEnvironmentWhenRefractoryPeriod()



