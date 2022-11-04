#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Traking
@File    : experimentPredtctOneTarget.py
@Author  : jay.zhu
@Time    : 2022/10/28 18:09
"""
import random
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from Scene.UAV_Scene.UAV_PredictTargetScene import UAV_PredictTargetScene
from MAS.Agents.UAV_Agent import UAV_Agent, UAV_TargetAgent
from EC.dynamicOpt.EC_DynamicOpt_HyperMutation import EC_DynamicOpt_HyperMutation
from EC.dynamicOpt.EC_DynamicOpt_InitAndHyperMutation import EC_DynamicOpt_InitAndHyperMutation
from MAS.MultiAgentSystem.UAV_MAS import UAV_PredictMAS


def experimentBase():
    AGENTS_NUM = 1
    AGENT_CLS = UAV_Agent.UAV_Agent
    OPTIMIZER_CLS = EC_DynamicOpt_InitAndHyperMutation
    TARGET_CLS = UAV_TargetAgent.UAV_TargetAgent
    MAS_CLS = UAV_PredictMAS.UAV_PredictMAS

    DELTA_TIME = .5
    AGENT_INIT_POSITION_RANGE = [[10., 15.], [10., 15.]]
    PREDICT_VELOCITY_LEN = 3
    USE_PREDICT_VELOCITY_LEN = 1
    AGENT_SELF_INIT_ARGS = {
        "initPositionState": [random.uniform(AGENT_INIT_POSITION_RANGE[0][0], AGENT_INIT_POSITION_RANGE[0][1]),
                              random.uniform(AGENT_INIT_POSITION_RANGE[1][0], AGENT_INIT_POSITION_RANGE[1][1]),
                              0],
        "linearVelocityRange":[0., 15.],
        "angularVelocityRange":[-200., 200.],
        "deltaTime":DELTA_TIME,
    }
    AGENT_SELF_INIT_ARGS_LIST = [{
        "initPositionState": [random.uniform(AGENT_INIT_POSITION_RANGE[0][0], AGENT_INIT_POSITION_RANGE[0][1]),
                              random.uniform(AGENT_INIT_POSITION_RANGE[1][0], AGENT_INIT_POSITION_RANGE[1][1]),
                              0],
        "linearVelocityRange":[0., 15.],
        "angularVelocityRange":[-200., 200.],
        "deltaTime":DELTA_TIME,
    }   for i in range(AGENTS_NUM)]
    AGENT_COMPUTATION_ARGS = {
        "predictVelocityLen": PREDICT_VELOCITY_LEN,
        "usePredictVelocityLen": USE_PREDICT_VELOCITY_LEN,
        "sameBestFittingCountThreshold": 10,
        "fittingIsSameThreshold": 1e-4,
        "JTaskFactor": 1.,
        "JConFactor": 1.,
        "JColFactor": 1.,
        "JComFactor": 1.,
    }
    AGENT_ARGS = {
        "initArgs":AGENT_SELF_INIT_ARGS_LIST,
        "computationArgs":AGENT_COMPUTATION_ARGS,
    }
    EC_INIT_ARGS = {
        "n":50,
        "dimNum":2,
        "needEpochTimes":100
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
        "borders":[0,1],
    }
    OPTIMIZER_ARGS = {
        "optimizerInitArgs":EC_INIT_ARGS,
        "optimizerComputationArgs":EC_COMPUTATION_ARGS
    }

    TARGET_INIT_POSITION_RANGE = [[10., 11.], [10., 11.]]
    TARGET_ARGS = {
        "initPositionState": [random.uniform(TARGET_INIT_POSITION_RANGE[0][0], TARGET_INIT_POSITION_RANGE[0][1]),
                              random.uniform(TARGET_INIT_POSITION_RANGE[1][0], TARGET_INIT_POSITION_RANGE[1][1]),
                              0],
        "linearVelocityRange": [0., 10.],
        "angularVelocityRange": [-30., 30.],
        "movingFuncRegister":"movingAsSin",
        "deltaTime": DELTA_TIME,
    }
    MAS_ARGS = {
        "optimizationNeedTimes": 1,
        "allCountDiffNashBalanceValue": 5e-1,
        "oneDiffNashBalanceValue": 1e-4,
        "predictVelocityLen": PREDICT_VELOCITY_LEN,
        "usePredictVelocityLen": USE_PREDICT_VELOCITY_LEN,
        "waitingInitPredictorTime":0,
    }

    NEED_RUNNING_TIME = 200


    uav_scene_base = UAV_PredictTargetScene(agentsNum=AGENTS_NUM,
                                    agentsCls=AGENT_CLS,
                                    agentsArgs=AGENT_ARGS,
                                    optimizerCls=OPTIMIZER_CLS,
                                    optimizerArgs=OPTIMIZER_ARGS,
                                    targetCls=TARGET_CLS,
                                    targetArgs=TARGET_ARGS,
                                    MAS_Cls=MAS_CLS,
                                    MAS_Args=MAS_ARGS,
                                    needRunningTime=NEED_RUNNING_TIME,
                                    deltaTime=DELTA_TIME)

    return uav_scene_base

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
    # experiment1()
    experimentTestDisBetweenUAVs()

if __name__ == "__main__":
    main()
