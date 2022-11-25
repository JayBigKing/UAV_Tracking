#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : experimentPredictMultiTarget.py
@Author  : jay.zhu
@Time    : 2022/11/5 15:32
"""
import sys
import time

sys.path.append("../../../")
import random
from experiment.experimentInit import experimentInit
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from Scene.UAV_Scene.multiTarget.UAV_MultiTarget_PredictScene import UAV_MultiTarget_PredictScene
from MAS.Agents.UAV_Agent import UAV_TargetAgent
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_Agent import UAV_MultiTarget_Agent
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_ProbabilitySelectTargetAgent import \
    UAV_MultiTargets_ProbabilitySelectTargetAgent
from EC.dynamicOpt.EC_DynamicOpt_InitAndHyperMutation import EC_DynamicOpt_InitAndHyperMutation
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_PredictMAS import UAV_MultiTarget_PredictMAS
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_PredictAndNashMAS import UAV_MultiTarget_PredictAndNashMAS
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_PredictAndSerialMAS import \
    UAV_MultiTarget_PredictAndSerialMAS


def experimentBase():
    AGENTS_NUM = 4
    TARGET_NUM = 2
    AGENT_CLS = UAV_MultiTargets_ProbabilitySelectTargetAgent
    OPTIMIZER_CLS = EC_DynamicOpt_InitAndHyperMutation
    TARGET_CLS = UAV_TargetAgent.UAV_TargetAgent
    MAS_CLS = UAV_MultiTarget_PredictMAS

    DELTA_TIME = .5
    AGENT_INIT_POSITION_RANGE = [[10., 20.], [10., 30.]]
    PREDICT_VELOCITY_LEN = 3
    USE_PREDICT_VELOCITY_LEN = 1
    MIN_DISTANCE_BETWEEN_UAV_THRESHOLD = 10.

    AGENT_SELF_INIT_ARGS_LIST = [{
        "initPositionState": [random.uniform(AGENT_INIT_POSITION_RANGE[0][0], AGENT_INIT_POSITION_RANGE[0][1]),
                              random.uniform(AGENT_INIT_POSITION_RANGE[1][0], AGENT_INIT_POSITION_RANGE[1][1]),
                              0],
        "linearVelocityRange": [0., 15.],
        "angularVelocityRange": [-200., 200.],
        "deltaTime": DELTA_TIME,
    } for i in range(AGENTS_NUM)]
    AGENT_COMPUTATION_ARGS = {
        "predictVelocityLen": PREDICT_VELOCITY_LEN,
        "usePredictVelocityLen": USE_PREDICT_VELOCITY_LEN,
        "sameBestFittingCountThreshold": 10,
        "fittingIsSameThreshold": 1e-4,
        "JTaskFactor": .4,
        "JConFactor": .0,
        "JColFactor": .4,
        "JComFactor": 1.,
        "JBalanceFactor": .4,
        "minDistanceThreshold": MIN_DISTANCE_BETWEEN_UAV_THRESHOLD,
    }
    AGENT_ARGS = {
        "initArgs": AGENT_SELF_INIT_ARGS_LIST,
        "computationArgs": AGENT_COMPUTATION_ARGS,
    }
    EC_INIT_ARGS = {
        "n": 50,
        "dimNum": 2,
        "needEpochTimes": 100
    }
    EC_COMPUTATION_ARGS = {
        "floatMutationOperateArg": 0.3,
        "floatCrossoverAlpha": 0.5,
        "mutationProbability": 0.05,
        "fittingMinDenominator": 0.2,
        "mutationProbabilityWhenChange": 0.5,
        "mutationProbabilityWhenNormal": 0.05,
        "performanceThreshold": 3,
        "refractoryPeriodLength": 2,
        "borders": [0, 1],
    }
    OPTIMIZER_ARGS = {
        "optimizerInitArgs": EC_INIT_ARGS,
        "optimizerComputationArgs": EC_COMPUTATION_ARGS
    }

    TARGET_INIT_POSITION_RANGE = [[10., 100.], [10., 100.]]
    TARGET_ARGS_LIST = [{
        "initPositionState": [random.uniform(TARGET_INIT_POSITION_RANGE[0][0], TARGET_INIT_POSITION_RANGE[0][1]),
                              random.uniform(TARGET_INIT_POSITION_RANGE[1][0], TARGET_INIT_POSITION_RANGE[1][1]),
                              0],
        "linearVelocityRange": [0., 10.],
        "angularVelocityRange": [-30., 30.],
        "movingFuncRegister": "randMoving",
        "deltaTime": DELTA_TIME,
    } for i in range(AGENTS_NUM)]

    MAS_ARGS = {
        "optimizationNeedTimes": 1,
        "allCountDiffNashBalanceValue": 5e-1,
        "oneDiffNashBalanceValue": 1e-4,
        "predictVelocityLen": PREDICT_VELOCITY_LEN,
        "usePredictVelocityLen": USE_PREDICT_VELOCITY_LEN,
        "waitingInitPredictorTime": 0,
        "lowerBoundOfUAVDis": MIN_DISTANCE_BETWEEN_UAV_THRESHOLD,
        "upperBoundOfUAVDis": False
    }

    NEED_RUNNING_TIME = 40

    uav_scene_base = UAV_MultiTarget_PredictScene(agentsNum=AGENTS_NUM,
                                                  agentsCls=AGENT_CLS,
                                                  agentsArgs=AGENT_ARGS,
                                                  optimizerCls=OPTIMIZER_CLS,
                                                  optimizerArgs=OPTIMIZER_ARGS,
                                                  targetCls=TARGET_CLS,
                                                  targetArgs=TARGET_ARGS_LIST,
                                                  MAS_Cls=MAS_CLS,
                                                  MAS_Args=MAS_ARGS,
                                                  needRunningTime=NEED_RUNNING_TIME,
                                                  targetNum=TARGET_NUM,
                                                  deltaTime=DELTA_TIME,
                                                  figureSavePath="../../experimentRes/experimentPredictMultiTarget/%s_%s" % (
                                                  time.strftime("%Y.%m.%d", time.localtime()), MAS_CLS.__name__)
                                                  )

    return uav_scene_base


@clockTester
def experiment1():
    uav_scene_base = experimentBase()
    uav_scene_base.run()


@clockTester
def experimentTestDisBetweenUAVs():
    uav_scene_base = experimentBase()
    uav_scene_base.run()
    needRunningTime = uav_scene_base.needRunningTimes
    multiAgentSystem = uav_scene_base.multiAgentSystem
    multiAgentSystem.UAVDisStatMatrix[:, :, 0] = multiAgentSystem.UAVDisStatMatrix[:, :, 0] / needRunningTime
    if hasattr(multiAgentSystem, "UAVDisStatMatrix"):
        print('avg distance matrix is : %r' % multiAgentSystem.UAVDisStatMatrix)


def main():
    experimentInit()
    experiment1()
    # experimentTestDisBetweenUAVs()


if __name__ == "__main__":
    main()
