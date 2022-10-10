from EC.EC_Base import EC_Base,EC_OtimizeWay,EC_SelectType,EC_CodingType
from EC.EC_WithStat_Base import EC_WithStat_Base
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
import numpy as np
def evalFunc1(chromosome):
    return (chromosome[0] - 1) ** 2
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

def main():
    test2()

if __name__ == "__main__":
    main()