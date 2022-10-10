import numpy as np
def NashShouldContinue(nowEpoch, needEpochTimes, agentPredictFittingValue, lastAgentPredictFittingValue, epsilon):
    if nowEpoch >= needEpochTimes:
        return False
    if nowEpoch > 50:
        for i in range(agentPredictFittingValue.shape[0]):
            if np.abs(agentPredictFittingValue[i] - lastAgentPredictFittingValue[i]) > epsilon:
                return True

        return False
    else:
        return True

def NashBalance(agents, TargetPredictTrajectory, needEpochTimes, epsilon):
    numOfAgents = len(agents)
    AgentsX = np.zeros((3, numOfAgents))
    bestAgentPredictU = np.zeros((agents[0].dimNum, numOfAgents))
    bestAgentPredictFittingValue = np.array([np.finfo(np.float32).max / 10. for i in range(numOfAgents)])
    nowAgentPredictFittingValue = np.array([np.finfo(np.float32).max  for i in range(numOfAgents)])
    nowEpoch = 0
    for i in range(numOfAgents):
        agents[i].predictPreInit()
        agents[i].predictTarget(TargetPredictTrajectory)
        bestAgentPredictU[:, i] = agents[i].EC_PredictArray[:]

    while NashShouldContinue(nowEpoch, needEpochTimes, nowAgentPredictFittingValue, bestAgentPredictFittingValue, epsilon) is True:
        for index, item in enumerate(agents):
            if item.UAVTrack.cmpFitting(nowAgentPredictFittingValue[index], bestAgentPredictFittingValue[index]) > 0:
                bestAgentPredictFittingValue[index] = nowAgentPredictFittingValue[index]
                bestAgentPredictU[:, index] = item.EC_PredictArray[:]

        for index, item in enumerate(agents):
            item.getInfoFromOthers(numOfAgents, AgentsX, bestAgentPredictU, index)
            item.predictTarget(TargetPredictTrajectory)
            _, nowAgentPredictFittingValue[index] = item.computing(nowEpoch)

        nowEpoch += 1


# import numpy as np
# def NashShouldContinue(agentPredictFittingValue, lastAgentPredictFittingValue, epsilon):
#     for i in range(agentPredictFittingValue.shape[0]):
#         if np.abs(agentPredictFittingValue[i] - lastAgentPredictFittingValue[i]) > epsilon:
#             return True
#
#     return False
#
# def NashBalance(agents, TargetPredictTrajectory, needEpochTimes, epsilon):
#     numOfAgents = len(agents)
#     AgentsX = np.zeros((3, numOfAgents))
#     # lastAgentPredictU = np.zeros((agents[0].dimNum, numOfAgents))
#     agentPredictU = np.zeros((agents[0].dimNum, numOfAgents))
#     lastAgentPredictFittingValue = np.array([-10. for i in range(numOfAgents)])
#     agentPredictFittingValue = np.zeros(numOfAgents)
#     for i in range(numOfAgents):
#         agents[i].predictPreInit()
#         agentPredictU[:, i] = agents[i].EC_PredictArray[:]
#
#     while NashShouldContinue(agentPredictFittingValue, lastAgentPredictFittingValue, epsilon) is True:
#
#         for index, item in enumerate(agents):
#             item.getInfoFromOthers(numOfAgents, AgentsX, agentPredictU, index)
#             item.predictTarget(TargetPredictTrajectory)
#             _, agentPredictFittingValue[index] = item.predict(needEpochTimes)
#             agentPredictU[:, index] = item.EC_PredictArray[:]
#
#         for index, item in enumerate(agentPredictFittingValue):
#             lastAgentPredictFittingValue[index] = item

# import numpy as np
# def NashShouldContinue(agentPredictFittingValue, lastAgentPredictFittingValue, epsilon):
#     for i in range(agentPredictFittingValue.shape[0]):
#         if np.abs(agentPredictFittingValue[i] - lastAgentPredictFittingValue[i]) > epsilon:
#             return True
#
#     return False
#
# def NashBalance(agents, TargetPredictTrajectory, needEpochTimes, epsilon):
#     numOfAgents = len(agents)
#     AgentsX = np.zeros((3, numOfAgents))
#     # lastAgentPredictU = np.zeros((agents[0].dimNum, numOfAgents))
#     bestAgentPredictU = np.zeros((agents[0].dimNum, numOfAgents))
#     bestAgentPredictFittingValue = np.array([np.finfo(np.float32).max / 10. for i in range(numOfAgents)])
#     nowAgentPredictFittingValue = np.array([np.finfo(np.float32).max  for i in range(numOfAgents)])
#     for i in range(numOfAgents):
#         agents[i].predictPreInit()
#         bestAgentPredictU[:, i] = agents[i].EC_PredictArray[:]
#         # lastAgentPredictFittingValue[i] = -1
#
#     k = 0
#     firstFlag = True
#     while NashShouldContinue(nowAgentPredictFittingValue, bestAgentPredictFittingValue, epsilon) is True:
#         k += 1
#         for index, item in enumerate(agents):
#             if item.UAVTrack.cmpFitting(nowAgentPredictFittingValue[index], bestAgentPredictFittingValue[index]) > 0:
#                 bestAgentPredictFittingValue[index] = nowAgentPredictFittingValue[index]
#                 bestAgentPredictU[:, index] = item.EC_PredictArray[:]
#
#         for index, item in enumerate(agents):
#             item.getInfoFromOthers(numOfAgents, AgentsX, bestAgentPredictU, index)
#             item.predictTarget(TargetPredictTrajectory)
#             if firstFlag is True:
#                 _, nowAgentPredictFittingValue[index] = item.predict(needEpochTimes)
#             else:
#                 _, nowAgentPredictFittingValue[index] = item.predict(needEpochTimes, doNotInitChromosome=True)
#
#         firstFlag = False


        # if k > 50:
        #     break

