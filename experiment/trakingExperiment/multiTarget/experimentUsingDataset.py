#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : experimentUsingDataset.py
@Author  : jay.zhu
@Time    : 2022/11/29 19:57
"""
import sys
import time

sys.path.append("../../../")
from experiment.experimentInit import experimentInit
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from experiment.datasetOperator import UAV_Tracking_DatasetOperator
from Scene.UAV_Scene.multiTarget.UAV_MultiTarget_UsingDatasetScene import UAV_MultiTarget_UsingDatasetScene
from MAS.Agents.UAV_Agent import UAV_Dataset_TargetAgent
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_ProbabilitySelectTargetAgent import \
    UAV_MultiTargets_ProbabilitySelectTargetAgent
from optimization.EC.dynamicOpt.EC_DynamicOpt_InitAndHyperMutation import EC_DynamicOpt_InitAndHyperMutation
from optimization.EC.dynamicOpt.EC_DynamicOpt_HMMemory import EC_DynamicOpt_HMMemory
from optimization.PSO.PSO_Tracking import PSO_Tracking
from optimization.EC.DiffEC.EC_DiffEC_Tracking_ADE import EC_DiffEC_Tracking_ADE
from optimization.EC.DiffEC.EC_DiffEC_Tracking_DE import EC_DiffEC_Tracking_DE
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_PredictMAS import UAV_MultiTarget_PredictMAS
from MAS.MultiAgentSystem.UAV_MAS.multiTarget.UAV_MultiTarget_PredictAndNashMAS import UAV_MultiTarget_PredictAndNashMAS

PSO_OPTIMIZATION_COMPUTATION_ARGS = {
    "fittingMinDenominator": 0.2,
    "w": 0.6,
    "c1": 2.,
    "c2": 0.3,
    "velocityFactor": 0.1,
    "borders": [0, 1],
}

ADE_OPTIMIZATION_COMPUTATION_ARGS = {
    "floatMutationOperateArg": 0.3,
    "floatCrossoverAlpha": 0.5,
    "mutationProbability": 0.05,
    "fittingMinDenominator": 0.2,

    "DiffCR0": 0.1,
    "DiffCR1": 0.6,
    "DiffF0": 0.1,
    "DiffF1": 0.6,
    "borders": [0, 1],
}

DYN_EC_OPTIMIZATION_COMPUTATION_ARGS = {
    "floatMutationOperateArg": 0.3,
    "floatCrossoverAlpha": 0.5,
    "mutationProbability": 0.05,
    "fittingMinDenominator": 0.2,
    "mutationProbabilityWhenChange": 0.6,
    "mutationProbabilityWhenNormal": 0.1,
    "performanceThreshold": 3,
    "refractoryPeriodLength": 2,
    "borders": [0, 1],
}

OPTIMIZATION_AND_ARGS_DICT = {
    "dynEC": {
        "class": EC_DynamicOpt_HMMemory,
        "computationArgs": DYN_EC_OPTIMIZATION_COMPUTATION_ARGS,
    },
    "ADE": {
        "class": EC_DiffEC_Tracking_ADE,
        "computationArgs": ADE_OPTIMIZATION_COMPUTATION_ARGS,
    },
    "DE": {
        "class": EC_DiffEC_Tracking_DE,
        "computationArgs": ADE_OPTIMIZATION_COMPUTATION_ARGS,
    },
    "PSO": {
        "class": PSO_Tracking,
        "computationArgs": PSO_OPTIMIZATION_COMPUTATION_ARGS,
    }
}

MAS_AND_ARGS_DICT = {
    "PredictMAS": {
        "class": UAV_MultiTarget_PredictMAS,
        "needTimes": 1,
    },
    "PredictAndNashMAS": {
        "class": UAV_MultiTarget_PredictAndNashMAS,
        "needTimes": 30,
    }
}

NEED_RUNNING_TIME = 3


def generateDataset():
    AGENTS_NUM = 4
    TARGET_NUM = 2
    AGENT_INIT_POSITION_RANGE = [[10., 20.], [10., 30.], [0., 0.]]
    TARGET_INIT_POSITION_RANGE = [[10., 100.], [10., 100.], [0., 0.]]
    TARGET_MOVING_WAY = "randMoving"

    datasetGenerator = UAV_Tracking_DatasetOperator.UAV_Tracking_DatasetGenerator()
    datasetGenerator.generateDataset(agentNum=AGENTS_NUM, targetNum=TARGET_NUM,
                                     movingTimes=NEED_RUNNING_TIME,
                                     agentInitPosRange=AGENT_INIT_POSITION_RANGE,
                                     targetInitPosRange=TARGET_INIT_POSITION_RANGE,
                                     targetMovingWay=TARGET_MOVING_WAY,
                                     targetLinearVelocityRange=[0., 10.],
                                     targetAngularVelocityRange=[-30., 30.])

def experimentBase(datasetPath, optimizerKey):
    AGENT_CLS = UAV_MultiTargets_ProbabilitySelectTargetAgent
    TARGET_CLS = UAV_Dataset_TargetAgent.UAV_Dataset_TargetAgent
    OPTIMIZER_KEY = optimizerKey
    MAS_KEY = "PredictAndNashMAS"

    DELTA_TIME = .5
    PREDICT_VELOCITY_LEN = 3
    USE_PREDICT_VELOCITY_LEN = 1
    MIN_DISTANCE_BETWEEN_UAV_THRESHOLD = 10.

    AGENT_SELF_INIT_ARGS_LIST = {
        "initPositionState": [],
        "linearVelocityRange": [0., 15.],
        "angularVelocityRange": [-200., 200.],
        "deltaTime": DELTA_TIME,
    }
    AGENT_COMPUTATION_ARGS = {
        "predictVelocityLen": PREDICT_VELOCITY_LEN,
        "usePredictVelocityLen": USE_PREDICT_VELOCITY_LEN,
        "sameBestFittingCountThreshold": 10,
        "fittingIsSameThreshold": 1e-4,
        "JTaskFactor": .4,
        "JConFactor": .0,
        "JColFactor": .8,
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
    EC_COMPUTATION_ARGS = OPTIMIZATION_AND_ARGS_DICT[OPTIMIZER_KEY]["computationArgs"]
    OPTIMIZER_ARGS = {
        "optimizerInitArgs": EC_INIT_ARGS,
        "optimizerComputationArgs": EC_COMPUTATION_ARGS
    }

    MAS_ARGS = {
        "optimizationNeedTimes": MAS_AND_ARGS_DICT[MAS_KEY]["needTimes"],
        "allCountDiffNashBalanceValue": 5e-1,
        "oneDiffNashBalanceValue": 1e-4,
        "predictVelocityLen": PREDICT_VELOCITY_LEN,
        "usePredictVelocityLen": USE_PREDICT_VELOCITY_LEN,
        "waitingInitPredictorTime": 0,
        "lowerBoundOfUAVDis": MIN_DISTANCE_BETWEEN_UAV_THRESHOLD,
        "upperBoundOfUAVDis": False
    }

    datasetLoader = UAV_Tracking_DatasetOperator.UAV_Tracking_DatasetLoader()
    UAV_Dataset = datasetLoader.loadDataset(datasetPath)

    uav_scene_base = UAV_MultiTarget_UsingDatasetScene(
        UAV_Dataset=UAV_Dataset,
        agentsCls=AGENT_CLS,
        agentsArgs=AGENT_ARGS,
        optimizerCls=OPTIMIZATION_AND_ARGS_DICT[OPTIMIZER_KEY]["class"],
        optimizerArgs=OPTIMIZER_ARGS,
        targetCls=TARGET_CLS,
        MAS_Cls=MAS_AND_ARGS_DICT[MAS_KEY]["class"],
        MAS_Args=MAS_ARGS,
        needRunningTime=NEED_RUNNING_TIME,
        deltaTime=DELTA_TIME,
        figureSavePath="../../experimentRes/experimentUsingDataset/%s_%s_%s" % (
            time.strftime("%Y.%m.%d", time.localtime()), MAS_AND_ARGS_DICT[MAS_KEY]["class"].__name__,
            OPTIMIZATION_AND_ARGS_DICT[OPTIMIZER_KEY]["class"].__name__,),
        userStatOutputRegisters=["UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE",
                                 "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STORE",
                                 "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE",
                                 "UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_STORE",
                                 "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_STORE",
                                 "UAV_MULTI_TARGET_SCENE_BASE_UAVAlertDisStore",
                                 "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore",
                                 "UAV_MULTI_TARGET_SCENE_BASE_TRACK_TARGET_ID_STORE",
                                 "UAV_MULTI_TARGET_SCENE_BASE_AVG_CLOSE_DIS_STORE"]
    )

    return uav_scene_base


DATASET_PATH = "targetTrajectoryDataset/randTarget2.json"


@clockTester
def experiment1Inner(optimizerKey):
    uav_scene_base = experimentBase(DATASET_PATH, optimizerKey)
    uav_scene_base.run()


@clockTester
def experiment1():
    optimizerKey = "dynEC"
    experiment1Inner(optimizerKey)

@clockTester
def experiment2():
    optimizerKeyList = OPTIMIZATION_AND_ARGS_DICT.keys()
    for optimizerKey in optimizerKeyList:
        experiment1Inner(optimizerKey)


def experiment():
    experimentInit()
    # generateDataset()
    experiment1()


if __name__ == "__main__":
    experiment()
