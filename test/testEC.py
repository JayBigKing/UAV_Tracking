import random
import numpy as np
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
from optimization.common.optimizationCommonEnum import OptimizationWay
from optimization.EC.EC_Base import EC_Base
from optimization.EC.EC_WithStat_Base import EC_WithStat_Base
from optimization.EC.DiffEC.EC_DiffEC_ADE import EC_DiffEC_ADE
from optimization.EC.dynamicOpt.EC_DynamicOpt_HyperMutation import EC_DynamicOpt_HyperMutation


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

def evalFunc3(solution):
    return np.sin(solution[0]) + 1

@clockTester
def test1():
    eb = EC_Base(100, 1, [100.], [-100.], evalFunc1, OptimizationWay.MAX ,100, {"borders":[1]})
    chromosome, bestVal = eb.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))

@clockTester
def test2():
    eb = EC_WithStat_Base(10, 1, [100.], [-100.], evalFunc1, OptimizationWay.MIN ,30, {"borders":[1]}, ["bestOverGen"])
    chromosome, bestVal = eb.optimize()
    avgOfBestChromosomesVal, bestChromosomesOverGen = eb.EC_WithStat_GetBestOverGen()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))
    print("[bestAimFuncVal bestFittingFuncVal]")
    print(avgOfBestChromosomesVal)
    print(bestChromosomesOverGen)

@clockTester
def test3():
    eb = EC_DynamicOpt_HyperMutation(10, 1, [100.], [-100.], evalFunc2, OptimizationWay.MIN ,
                                     needEpochTimes=100,
                                     ECArgs={"borders":[1], "performanceThreshold": 2., "refractoryPeriodLength":1},
                                     statRegisters=["bestOverGen"])
    chromosome, bestVal = eb.optimize()
    avgOfBestChromosomesVal, bestChromosomesOverGen = eb.EC_WithStat_GetBestOverGen()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))
    print("[bestAimFuncVal bestFittingFuncVal]")
    print(avgOfBestChromosomesVal)


@clockTester
def testADE():
    eb = EC_DiffEC_ADE(n=100, dimNum=1, maxConstraint=[100.], minConstraint=[-100.],
                       evalVars=evalFunc1, otimizeWay=OptimizationWay.MIN ,needEpochTimes=100, ECArgs={"borders":[1]})
    chromosome, bestVal = eb.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))

@clockTester
def testDynEC():
    eb = EC_DynamicOpt_HyperMutation(10, 1, [0.], [2 * np.pi], evalFunc3, OptimizationWay.MAX ,
                                     needEpochTimes=100,
                                     ECArgs={"borders":[1], "performanceThreshold": 2., "refractoryPeriodLength":1},
                                     statRegisters=["bestOverGen"])
    chromosome, bestVal = eb.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))

@clockTester
def testDynDE():
    from optimization.EC.dynamicOpt.DE.EC_DynamicOpt_DEMemory import EC_DynamicOpt_DEMemory
    eb = EC_DynamicOpt_DEMemory(10, 1, [0.], [2 * np.pi], evalFunc3, OptimizationWay.MIN ,
                                     needEpochTimes=100,
                                     ECArgs={"borders":[1], "performanceThreshold": 2., "refractoryPeriodLength":1},
                                     statRegisters=["bestOverGen"])

    chromosome, bestVal, _ = eb.optimize()
    print(f'best chromosome : {chromosome !r}\r\nbestVal : {bestVal !r}'.format(chromosome=chromosome, bestVal=bestVal))

def main():
    testDynDE()

if __name__ == "__main__":
    main()