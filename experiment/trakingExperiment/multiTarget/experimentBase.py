#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : experimentBase.py
@Author  : jay.zhu
@Time    : 2023/1/19 1:05
"""
import sys
import time

sys.path.append("../../../")
from experiment.experimentInit import experimentInit
from Jay_Tool.LogTool import myLogger
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from experiment.datasetOperator import UAV_Tracking_DatasetOperator
from Scene.UAV_Scene.multiTarget.UAV_MultiTarget_UsingDatasetScene import UAV_MultiTarget_UsingDatasetScene, \
    UAV_MultiTarget_PredictScene
from MAS.Agents.UAV_Agent import UAV_Dataset_TargetAgent
from MAS.Agents.UAV_Agent.multiTarget.UAV_MultiTargets_ProbabilitySelectTargetAgent import \
    UAV_MultiTargets_ProbabilitySelectTargetAgent
from optimization.EC.dynamicOpt.EC_DynamicOpt_InitAndHyperMutation import EC_DynamicOpt_InitAndHyperMutation
from optimization.EC.dynamicOpt.EC_DynamicOpt_HMMemory import EC_DynamicOpt_HMMemory
from optimization.PSO.PSO_Tracking import PSO_Tracking
from optimization.EC.DiffEC.EC_DiffEC_Tracking_ADE import EC_DiffEC_Tracking_ADE
from optimization.EC.DiffEC.EC_DiffEC_Tracking_DE import EC_DiffEC_Tracking_DE
from optimization.EC.EC_Tracking import EC_Tracking
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

NO_DYN_EC_OPTIMIZATION_COMPUTATION_ARGS = {
    "floatMutationOperateArg": 0.3,
    "floatCrossoverAlpha": 0.5,
    "mutationProbability": 0.05,
    "fittingMinDenominator": 0.2,

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
    },
    "noDynEC": {
        "class": EC_Tracking,
        "computationArgs": NO_DYN_EC_OPTIMIZATION_COMPUTATION_ARGS,
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

S_DEFAULT_DELTA_TIME = .5
S_DEFAULT_AGENT_INIT_POSITION_RANGE = [[10., 20.], [10., 30.]]
S_DEFAULT_PREDICT_VELOCITY_LEN = 3
S_DEFAULT_USE_PREDICT_VELOCITY_LEN = 1
S_DEFAULT_MIN_DISTANCE_BETWEEN_UAV_THRESHOLD = 10.
S_DEFAULT_AGENT_SELF_INIT_ARGS = {
    "initPositionState": [],
    "linearVelocityRange": [0., 15.],
    "angularVelocityRange": [-200., 200.],
    "deltaTime": S_DEFAULT_DELTA_TIME,
}
S_DEFAULT_AGENT_COMPUTATION_ARGS = {
    "predictVelocityLen": S_DEFAULT_PREDICT_VELOCITY_LEN,
    "usePredictVelocityLen": S_DEFAULT_USE_PREDICT_VELOCITY_LEN,
    "sameBestFittingCountThreshold": 10,
    "fittingIsSameThreshold": 1e-4,
    "JTaskFactor": .4,
    "JConFactor": .0,
    "JColFactor": .8,
    "JComFactor": 1.,
    "JBalanceFactor": .4,
    "minDistanceThreshold": S_DEFAULT_MIN_DISTANCE_BETWEEN_UAV_THRESHOLD,
}
S_DEFAULT_EC_INIT_ARGS = {
    "n": 50,
    "dimNum": 2,
    "needEpochTimes": 100
}
S_DEFAULT_MAS_ARGS = {
    "optimizationNeedTimes": 0,
    "allCountDiffNashBalanceValue": 5e-1,
    "oneDiffNashBalanceValue": 1e-4,
    "predictVelocityLen": S_DEFAULT_PREDICT_VELOCITY_LEN,
    "usePredictVelocityLen": S_DEFAULT_USE_PREDICT_VELOCITY_LEN,
    "waitingInitPredictorTime": 0,
    "lowerBoundOfUAVDis": S_DEFAULT_MIN_DISTANCE_BETWEEN_UAV_THRESHOLD,
    "upperBoundOfUAVDis": False
}
S_DEFAULT_TARGET_INIT_POSITION_RANGE = [[10., 100.], [10., 100.]]
S_DEFAULT_TARGET_ARGS = {
    "initPositionState": [],
    "linearVelocityRange": [0., 10.],
    "angularVelocityRange": [-30., 30.],
    "movingFuncRegister": "randMoving",
    "deltaTime": S_DEFAULT_DELTA_TIME,
}

S_DEFAULT_USER_STAT_OUTPUT_REGISTERS = ["UAV_MULTI_TARGET_SCENE_BASE_DIS_BETWEEN_TARGET_AND_UAV_STORE",
                                        "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STORE",
                                        "UAV_MULTI_TARGET_SCENE_BASE_AVG_DIS_STABILITY_STORE",
                                        "UAV_MULTI_TARGET_SCENE_BASE_TARGET_TRACKED_NUM_VARIANCE_STORE",
                                        "UAV_MULTI_TARGET_SCENE_BASE_EFFECTIVE_TIME_STORE",
                                        "UAV_MULTI_TARGET_SCENE_BASE_UAVAlertDisStore",
                                        "UAV_MULTI_TARGET_SCENE_BASE_UAVFitnessStore",
                                        "UAV_MULTI_TARGET_SCENE_BASE_TRACK_TARGET_ID_STORE",
                                        "UAV_MULTI_TARGET_SCENE_BASE_AVG_CLOSE_DIS_STORE"]
S_DEFAULT_SCENE_DEFAULT_ARGS = {
    "ifPrintRunningEpoch": True
}

NEED_RUNNING_TIME = 3


def generateDataset(agentNum = 4, targetNum = 2, targetMovingWay = "randMoving"):
    import os
    def getFileName():
        nameIndex = 1
        storeFileNameWithPathNoSubfix = "%s%s%d" % (STORE_PATH, targetMovingWay, nameIndex)
        if os.path.exists("%s.json" % storeFileNameWithPathNoSubfix):
            nameIndex += 1
            while True:
                storeFileNameWithPathNoSubfix = "%s%s%d" % (STORE_PATH, targetMovingWay, nameIndex)
                if os.path.exists("%s.json" % storeFileNameWithPathNoSubfix):
                    nameIndex += 1
                else:
                    return "%s%d.json" % (targetMovingWay, nameIndex)
        else:
            return "%s%d.json" % (targetMovingWay, nameIndex)

    AGENTS_NUM = agentNum
    TARGET_NUM = targetNum
    AGENT_INIT_POSITION_RANGE = [[10., 20.], [10., 30.], [0., 0.]]
    TARGET_INIT_POSITION_RANGE = [[10., 100.], [10., 100.], [0., 0.]]
    TARGET_MOVING_WAY = targetMovingWay
    STORE_PATH = "./targetTrajectoryDataset/"
    fileName = getFileName()

    datasetGenerator = UAV_Tracking_DatasetOperator.UAV_Tracking_DatasetGenerator()
    datasetGenerator.generateDataset(agentNum=AGENTS_NUM, targetNum=TARGET_NUM,
                                     movingTimes=NEED_RUNNING_TIME,
                                     agentInitPosRange=AGENT_INIT_POSITION_RANGE,
                                     targetInitPosRange=TARGET_INIT_POSITION_RANGE,
                                     targetMovingWay=TARGET_MOVING_WAY,
                                     targetLinearVelocityRange=[0., 10.],
                                     targetAngularVelocityRange=[-30., 30.],
                                     fileName = fileName,
                                     storePath=STORE_PATH)

    return "%s%s" % (STORE_PATH, fileName)


def experimentBaseCheckInput(agentCls, getTargetTrajectoryWay, masKey, optimizerKey,
                             agentsNum=None, targetNum=None, datasetPath=None,
                             agentSelfInitArgsList=None, agentComputationArgs=None,
                             ECInitArgs=None, targetArgsList=None, targetMovingWay=None,
                             masArgs=None, figureSavePathKeyword=None, userStatOutputRegisters=None,
                             sceneArgs=None
                             ):
    import random
    from optimization.EC.EC_Base import EC_Base
    from optimization.PSO.PSO_Base import PSO_Base
    from MAS.Agents.UAV_Agent.Agent_UAV_Base import Agent_UAV_Base
    optimizerWhiteList = {EC_Base, PSO_Base}
    fmtOfTypeError = "The variable {0:s} must be {1:s}"
    fmtOfTypeErrorForSubClass = "The variable {0:s} must be subclass of {1:s}"
    try:
        AGENT_CLS = agentCls
        # if isinstance(agentCls, Agent_UAV_Base):
        #     AGENT_CLS = UAV_MultiTargets_ProbabilitySelectTargetAgent
        # else:
        #     raise TypeError(fmtOfTypeErrorForSubClass.format("agentCls", "Agent_UAV_Base"))

        if isinstance(getTargetTrajectoryWay, str) is False:
            raise ValueError(fmtOfTypeError.format("getTargetTrajectoryWay", "str"))
        if getTargetTrajectoryWay == "online":
            TARGET_CLS = UAV_Dataset_TargetAgent.UAV_TargetAgent
        elif getTargetTrajectoryWay == "dataset":
            TARGET_CLS = UAV_Dataset_TargetAgent.UAV_Dataset_TargetAgent
        else:
            raise ValueError("The variable getTargetTrajectoryWay must be enum str: \" online \" or \" dataset \"")

        # inWhiteList = False
        # for item in optimizerWhiteList:
        #     if isinstance(optimizerKey, item):
        #         inWhiteList = True
        #         break
        # if inWhiteList is True:
        #     OPTIMIZER_KEY = optimizerKey
        # else:
        #     fmt = "optimizerKey must be subclass of "
        #     for item in optimizerWhiteList:
        #         fmt += "%s , " % (item.__name__)
        #     raise ValueError(fmt[0: fmt.rfind(" , ")])
        OPTIMIZER_KEY = optimizerKey

        if isinstance(masKey, str) is False:
            raise ValueError(fmtOfTypeError.format("masKey", "str"))
        elif masKey in MAS_AND_ARGS_DICT.keys():
            MAS_KEY = masKey
            MAS_CLS = MAS_AND_ARGS_DICT[masKey]["class"]
        else:
            fmt = "masKey must be str of "
            for item in MAS_AND_ARGS_DICT.keys():
                fmt += "%s , " % (item)
            raise ValueError(fmt[0: fmt.rfind(" , ")])

        if agentSelfInitArgsList is not None:
            AGENT_SELF_INIT_ARGS_LIST = agentSelfInitArgsList
        else:
            if getTargetTrajectoryWay == "online":
                AGENT_SELF_INIT_ARGS_LIST = [dict(S_DEFAULT_AGENT_SELF_INIT_ARGS) for i in range(agentsNum)]
                for i in range(agentsNum):
                    AGENT_SELF_INIT_ARGS_LIST[i]["initPositionState"] = [
                        random.uniform(S_DEFAULT_AGENT_INIT_POSITION_RANGE[0][0],
                                       S_DEFAULT_AGENT_INIT_POSITION_RANGE[0][1]),
                        random.uniform(S_DEFAULT_AGENT_INIT_POSITION_RANGE[1][0],
                                       S_DEFAULT_AGENT_INIT_POSITION_RANGE[1][1]),
                        0]

            else:
                AGENT_SELF_INIT_ARGS_LIST = dict(S_DEFAULT_AGENT_SELF_INIT_ARGS)

        if agentComputationArgs is not None:
            AGENT_COMPUTATION_ARGS = agentComputationArgs
        else:
            AGENT_COMPUTATION_ARGS = dict(S_DEFAULT_AGENT_COMPUTATION_ARGS)

        AGENT_ARGS = {
            "initArgs": AGENT_SELF_INIT_ARGS_LIST,
            "computationArgs": AGENT_COMPUTATION_ARGS,
        }

        if ECInitArgs is not None:
            EC_INIT_ARGS = ECInitArgs
        else:
            EC_INIT_ARGS = dict(S_DEFAULT_EC_INIT_ARGS)

        EC_COMPUTATION_ARGS = OPTIMIZATION_AND_ARGS_DICT[OPTIMIZER_KEY]["computationArgs"]
        OPTIMIZER_ARGS = {
            "optimizerInitArgs": EC_INIT_ARGS,
            "optimizerComputationArgs": EC_COMPUTATION_ARGS
        }

        if masArgs is not None:
            MAS_ARGS = masArgs
        else:
            MAS_ARGS = dict(S_DEFAULT_MAS_ARGS)
            MAS_ARGS["optimizationNeedTimes"] = MAS_AND_ARGS_DICT[MAS_KEY]["needTimes"]

        if getTargetTrajectoryWay == "dataset":
            datasetLoader = UAV_Tracking_DatasetOperator.UAV_Tracking_DatasetLoader()
            UAV_Dataset = datasetLoader.loadDataset(datasetPath)
            TARGET_ARGS_LIST = []
        else:
            UAV_Dataset = []
            if targetArgsList is not None:
                TARGET_ARGS_LIST = targetArgsList
            else:
                if isinstance(targetMovingWay, str) is False:
                    raise TypeError(fmtOfTypeError.format("targetMovingWay", "str"))
                TARGET_ARGS_LIST = [dict(S_DEFAULT_TARGET_ARGS) for i in range(targetNum)]
                for i in range(targetNum):
                    TARGET_ARGS_LIST[i]["initPositionState"] = [
                        random.uniform(S_DEFAULT_TARGET_INIT_POSITION_RANGE[0][0],
                                       S_DEFAULT_TARGET_INIT_POSITION_RANGE[0][1]),
                        random.uniform(S_DEFAULT_TARGET_INIT_POSITION_RANGE[1][0],
                                       S_DEFAULT_TARGET_INIT_POSITION_RANGE[1][1]),
                        0]
                    TARGET_ARGS_LIST[i]["movingFuncRegister"] = targetMovingWay

        if figureSavePathKeyword is None:
            FIGURE_SAVE_PATH = "../../experimentRes/experimentBase/%s_%s_%s" % (
                time.strftime("%Y.%m.%d", time.localtime()), MAS_AND_ARGS_DICT[MAS_KEY]["class"].__name__,
                OPTIMIZATION_AND_ARGS_DICT[OPTIMIZER_KEY]["class"].__name__,)
        else:
            FIGURE_SAVE_PATH = figureSavePathKeyword

        if userStatOutputRegisters is None:
            USER_STAT_OUTPUT_REGISTERS = list(S_DEFAULT_USER_STAT_OUTPUT_REGISTERS)
        else:
            USER_STAT_OUTPUT_REGISTERS = userStatOutputRegisters

        if sceneArgs is None:
            SCENE_ARGS = dict(S_DEFAULT_SCENE_DEFAULT_ARGS)
        else:
            SCENE_ARGS = sceneArgs

        return {
            "AGENT_CLS": AGENT_CLS,
            "TARGET_CLS": TARGET_CLS,
            "OPTIMIZER_CLS": OPTIMIZATION_AND_ARGS_DICT[OPTIMIZER_KEY]["class"],
            "MAS_CLS": MAS_CLS,
            "AGENT_ARGS": AGENT_ARGS,
            "OPTIMIZER_ARGS": OPTIMIZER_ARGS,
            "MAS_ARGS": MAS_ARGS,
            "UAV_Dataset": UAV_Dataset,
            "TARGET_ARGS_LIST": TARGET_ARGS_LIST,
            "FIGURE_SAVE_PATH": FIGURE_SAVE_PATH,
            "USER_STAT_OUTPUT_REGISTERS": USER_STAT_OUTPUT_REGISTERS,
            "SCENE_ARGS": SCENE_ARGS
        }

    except ValueError as e:
        myLogger.myLogger_Logger().error(repr(e))
    except TypeError as e:
        myLogger.myLogger_Logger().error(repr(e))


def experimentBase(agentCls, getTargetTrajectoryWay, masKey, optimizerKey, agentsNum=None, targetNum=None,
                   datasetPath=None, agentSelfInitArgsList=None, agentComputationArgs=None, ECInitArgs=None,
                   targetArgsList=None, targetMovingWay=None, masArgs=None, uavSceneCls=None,
                   figureSavePathKeyword=None, userStatOutputRegisters=None, sceneArgs=None):

    argDict = experimentBaseCheckInput(agentCls, getTargetTrajectoryWay, masKey, optimizerKey,
                                       agentsNum=agentsNum,
                                       targetNum=targetNum,
                                       datasetPath=datasetPath,
                                       agentSelfInitArgsList=agentSelfInitArgsList,
                                       agentComputationArgs=agentComputationArgs,
                                       ECInitArgs=ECInitArgs,
                                       targetArgsList=targetArgsList,
                                       targetMovingWay=targetMovingWay,
                                       masArgs=masArgs,
                                       figureSavePathKeyword=figureSavePathKeyword,
                                       userStatOutputRegisters=userStatOutputRegisters,
                                       sceneArgs=sceneArgs,)
    if getTargetTrajectoryWay == "online":
        if uavSceneCls is not None:
            uav_scene = uavSceneCls(agentsNum=agentsNum,
                                                     agentsCls=argDict["AGENT_CLS"],
                                                     agentsArgs=argDict["AGENT_ARGS"],
                                                     optimizerCls=argDict["OPTIMIZER_CLS"],
                                                     optimizerArgs=argDict["OPTIMIZER_ARGS"],
                                                     targetCls=argDict["TARGET_CLS"],
                                                     targetArgs=argDict["TARGET_ARGS_LIST"],
                                                     MAS_Cls=argDict["MAS_CLS"],
                                                     MAS_Args=argDict["MAS_ARGS"],
                                                     needRunningTime=NEED_RUNNING_TIME,
                                                     targetNum=targetNum,
                                                     deltaTime=S_DEFAULT_DELTA_TIME,
                                                     figureSavePath=argDict["FIGURE_SAVE_PATH"],
                                                     userStatOutputRegisters=argDict["USER_STAT_OUTPUT_REGISTERS"],
                                                     sceneArgs=argDict["SCENE_ARGS"])
        else:
            uav_scene = UAV_MultiTarget_PredictScene(agentsNum=agentsNum,
                                                     agentsCls=argDict["AGENT_CLS"],
                                                     agentsArgs=argDict["AGENT_ARGS"],
                                                     optimizerCls=argDict["OPTIMIZER_CLS"],
                                                     optimizerArgs=argDict["OPTIMIZER_ARGS"],
                                                     targetCls=argDict["TARGET_CLS"],
                                                     targetArgs=argDict["TARGET_ARGS_LIST"],
                                                     MAS_Cls=argDict["MAS_CLS"],
                                                     MAS_Args=argDict["MAS_ARGS"],
                                                     needRunningTime=NEED_RUNNING_TIME,
                                                     targetNum=targetNum,
                                                     deltaTime=S_DEFAULT_DELTA_TIME,
                                                     figureSavePath=argDict["FIGURE_SAVE_PATH"],
                                                     userStatOutputRegisters=argDict["USER_STAT_OUTPUT_REGISTERS"],
                                                     sceneArgs=argDict["SCENE_ARGS"])
    else:
        if uavSceneCls is not None:
            uav_scene = uavSceneCls(
                UAV_Dataset=argDict["UAV_Dataset"],
                agentsCls=argDict["AGENT_CLS"],
                agentsArgs=argDict["AGENT_ARGS"],
                optimizerCls=argDict["OPTIMIZER_CLS"],
                optimizerArgs=argDict["OPTIMIZER_ARGS"],
                targetCls=argDict["TARGET_CLS"],
                MAS_Cls=argDict["MAS_CLS"],
                MAS_Args=argDict["MAS_ARGS"],
                needRunningTime=NEED_RUNNING_TIME,
                deltaTime=S_DEFAULT_DELTA_TIME,
                figureSavePath=argDict["FIGURE_SAVE_PATH"],
                userStatOutputRegisters=argDict["USER_STAT_OUTPUT_REGISTERS"],
                sceneArgs=argDict["SCENE_ARGS"]
            )
        else:
            uav_scene = UAV_MultiTarget_UsingDatasetScene(
                UAV_Dataset=argDict["UAV_Dataset"],
                agentsCls=argDict["AGENT_CLS"],
                agentsArgs=argDict["AGENT_ARGS"],
                optimizerCls=argDict["OPTIMIZER_CLS"],
                optimizerArgs=argDict["OPTIMIZER_ARGS"],
                targetCls=argDict["TARGET_CLS"],
                MAS_Cls=argDict["MAS_CLS"],
                MAS_Args=argDict["MAS_ARGS"],
                needRunningTime=NEED_RUNNING_TIME,
                deltaTime=S_DEFAULT_DELTA_TIME,
                figureSavePath=argDict["FIGURE_SAVE_PATH"],
                userStatOutputRegisters=argDict["USER_STAT_OUTPUT_REGISTERS"],
                sceneArgs=argDict["SCENE_ARGS"]
            )

    return uav_scene
