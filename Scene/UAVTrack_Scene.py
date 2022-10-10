import numpy as np
import random
from UAV import My_UAV
from Target import Target
from MAS.NashBalance import NashBalance
from TargetMovePredictor import TargetMovePredictor
import sys

sys.path.append("../visualize")
from CoorDiagram import CoorDiagram


class UAVTrack_Scene:
    def __init__(self, numOfUAV, numOfChromsomes, totalTime, timeslotLength, domainTimeLeght, UAVHeight,
                 UAVInitX_Range, UAVInitY_Range, TargetInitX_Range, TargetInitY_Range, TargetVRange, TargetAngleRange,
                 maxConstraint, minConstraint, fittingLambdaList, F0, F1, CR0, CR1, dmin, dmax, deltaD, needEpochTimes,
                 NashEpsilon):
        self.init(numOfUAV, numOfChromsomes, totalTime, timeslotLength, domainTimeLeght, UAVHeight,
                  UAVInitX_Range, UAVInitY_Range, TargetInitX_Range, TargetInitY_Range, TargetVRange, TargetAngleRange,
                  maxConstraint, minConstraint, fittingLambdaList, F0, F1, CR0, CR1, dmin, dmax, deltaD, needEpochTimes,
                  NashEpsilon)

    def init(self, numOfUAV, numOfChromsomes, totalTime, timeslotLength, domainTimeLength, UAVHeight,
             UAVInitX_Range, UAVInitY_Range, TargetInitX_Range, TargetInitY_Range, TargetVRange, TargetAngleRange,
             maxConstraint, minConstraint, fittingLambdaList, F0, F1, CR0, CR1, dmin, dmax, deltaD, needEpochTimes,
             NashEpsilon):
        # initX, height, n, timeStepsQ, maxConstraint, minConstraint, fittingLambdaList, F0, F1, CR0, CR1, dmin,
        # dmax, deltaD, deltaTime = 1
        self.timeStepsQ = int(np.ceil(domainTimeLength / timeslotLength))
        self.timeslotLength = timeslotLength
        self.totalTime = totalTime
        self.uavTeam = [My_UAV([random.uniform(UAVInitX_Range[0], UAVInitX_Range[1]),
                                random.uniform(UAVInitY_Range[0], UAVInitY_Range[1])],
                               UAVHeight, numOfChromsomes, self.timeStepsQ, maxConstraint, minConstraint,
                               fittingLambdaList, F0, F1, CR0, CR1,
                               dmin, dmax, deltaD,
                               timeslotLength) for i in range(numOfUAV)]
        self.target = Target([random.uniform(TargetInitX_Range[0], TargetInitX_Range[1]),
                              random.uniform(TargetInitY_Range[0], TargetInitY_Range[1])],
                             speedRange=TargetVRange, angleRange=TargetAngleRange, deltaTime=timeslotLength
                             )
        self.targetMovePredictor = TargetMovePredictor(timeslotLength)

        self.argumentMap = {
            "F": {
                "F0": F0,
                "F1": F1
            },
            "CR": {
                "CR0": CR0,
                "CR1": CR1
            },
            "D": {
                "dmin": dmin,
                "dmax": dmax,
                "deltaD": deltaD
            },
            "ECArg": {
                "needEpochTimes": needEpochTimes,
                "fittingLambdaList": fittingLambdaList
            },
            "NashArg": {
                "NashEpsilon": NashEpsilon
            },
            "timeArg": {
                "totalTime": totalTime,
                "timeslotLength": timeslotLength,
                "domainTimeLength": domainTimeLength,
                "timeStepsQ": self.timeStepsQ
            }
        }

    def run(self):
        # nowTimeStep = 0
        # while nowTimeStep <= (self.argumentMap["timeArg"]["totalTime"] - self.argumentMap["timeArg"]["domainTimeLength"]) / self.argumentMap["timeArg"]["timeslotLength"]:
        #     self.targetMovePredictor.predict(self.target.randMoving())
        #     nowTimeStep += 1

        targetMovingScatter = [self.target.X[:2].tolist()]
        uavMovingScatters = [[item.X[:2].tolist()] for item in self.uavTeam]
        nowTime = 0.0
        while nowTime < self.totalTime:
            print(r'%f s / %d s of timestep' % (nowTime, self.totalTime))
            # targetMovingScatter.append(self.target.randMoving().tolist())
            targetMovingScatter.append(self.target.movingAsFunction())
            predictTargetXList = self.targetMovePredictor.predictMultiSet(targetMovingScatter[-1], self.argumentMap["timeArg"]["timeStepsQ"], deltaTime=self.timeslotLength)
            NashBalance(self.uavTeam, predictTargetXList, self.argumentMap["ECArg"]["needEpochTimes"],
                        self.argumentMap["NashArg"]["NashEpsilon"])
            for i in range(len(self.uavTeam)):
                self.uavTeam[i].movingUsingPredictList(0)
                uavMovingScatters[i].append(self.uavTeam[i].X[:2].tolist())
            nowTime += self.timeslotLength

        scattersList = [targetMovingScatter]
        nameList = ["target"]
        for i, item  in enumerate(uavMovingScatters):
            scattersList.append(item)
            nameList.append(r"uav %d" % i)
        cd = CoorDiagram()
        cd.drwaManyScattersInOnePlane(scattersList, nameList=nameList)
