import random

import numpy as np
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from EC.EC_Base import EC_Base,EC_OtimizeWay,EC_SelectType,EC_CodingType
from EC.EC_WithStat_Base import EC_WithStat_Base
from EC.dynamicOpt.EC_DynamicOpt_HyperMutation import EC_DynamicOpt_HyperMutation


def evalFunc1(chromosome):
    return (chromosome[0] - 1) ** 2

evalFunc2Count = 0
evalFunc2Val = 2.
def evalFunc2(chromosome):
    global evalFunc2Count
    global evalFunc2Val
    if evalFunc2Count == 0:
        evalFunc2Val = float(random.randint(1, 10))
        print('evalFunc2Val is %f'%evalFunc2Val)
    elif evalFunc2Count == 10 * 30:
        evalFunc2Val = float(random.randint(-60, -10))
        print('evalFunc2Val is %f'%evalFunc2Val)

    evalFunc2Count += 1

    return (2 * chromosome[0] - evalFunc2Val) ** 2

@clockTester
def test1():
    eb = EC_Base(100, 1, [100.], [-100.], evalFunc1, EC_OtimizeWay.MAX ,100, {"borders":[1]})
    chromosome, bestVal = eb.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))

@clockTester
def test2():
    eb = EC_WithStat_Base(10, 1, [100.], [-100.], evalFunc1, EC_OtimizeWay.MIN ,30, {"borders":[1]}, ["bestOverGen"])
    chromosome, bestVal = eb.optimize()
    avgOfBestChromosomesVal, bestChromosomesOverGen = eb.EC_WithStat_GetBestOverGen()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))
    print("[bestAimFuncVal bestFittingFuncVal]")
    print(avgOfBestChromosomesVal)
    print(bestChromosomesOverGen)

@clockTester
def test3():
    eb = EC_DynamicOpt_HyperMutation(10, 1, [100.], [-100.], evalFunc2, EC_OtimizeWay.MIN ,
                                     needEpochTimes=100,
                                     ECArgs={"borders":[1], "performanceThreshold": 2., "refractoryPeriodLength":1},
                                     statRegisters=["bestOverGen", "printOutEveryGen"])
    chromosome, bestVal = eb.optimize()
    avgOfBestChromosomesVal, bestChromosomesOverGen = eb.EC_WithStat_GetBestOverGen()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))
    print("[bestAimFuncVal bestFittingFuncVal]")
    print(avgOfBestChromosomesVal)



def main():
    test3()

if __name__ == "__main__":
    main()