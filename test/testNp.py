import numpy as np
import bisect
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester
def test1():
    np1 = np.array([float(i) for i in range(10)])
    np2 = np.zeros((10,2))
    for i in range(2):
        for j in range(10):
            np2[j][i] = float(j)
    print(np.sum(np1))
    print(np.sum(np2))
def test2():
    num = 20
    np1 = np.array([float(i) for i in range(1, num + 1)])
    totalFittingValue = np.sum(np1)
    fittingProbabilityCount = 0.
    fittingProbability = [np1[i] / totalFittingValue for i in range(num)]
    for i in range(num):
        fittingProbabilityCount += fittingProbability[i]
        fittingProbability[i] = fittingProbabilityCount
    print(fittingProbability)
    print(bisect.bisect_left(fittingProbability, 0.905))

def test3():
    num = 20
    np1 = np.array([1 / float(i) for i in range(1, num + 1)])
    totalFittingValue = np.sum(np1)
    fittingProbabilityCount = 0.
    fittingProbability = [np1[i] / totalFittingValue for i in range(num)]
    for i in range(num):
        fittingProbabilityCount += fittingProbability[i]
        fittingProbability[i] = fittingProbabilityCount
    print(fittingProbability)

def test4():
    list0 = [i for i in range(10)]
    list0.extend([i for i in range(10)])
    print(list0)

def test5():
    list0 = [i for i in range(10)]
    list0.extend([i for i in range(10)])
    print(list0[1:2] + list0[5:10])

def test6():
    np2 = np.zeros((10,2))
    for i in range(2):
        for j in range(10):
            np2[j][i] = float(j)
    np3 = np.vstack(np.sum(np2, axis=1))

    print(np3)
    print(np.square(np3))
    print(np2 - np3)
    print(np2)
    print(np.sum(np.square(np2 - np3)))

def test7():
    n0 = np.array([i + 10 for i in range(20)])
    l0 = [0, 4, 5]
    print(n0.take(l0))
    print(np.average(n0))
    print(np.average(n0.take(l0)))
    print(len(n0))

def test8():
    n0 = np.array([i + 10 for i in range(20)])
    for item in n0:
        print(item)

def test9():
    n0 = np.array([i + 10 for i in range(20)])
    n1 = n0
    n0[2]= 0.
    for item in n1:
        print(item)

def test10():
    n0 = np.array([i + 10 for i in range(20)])
    n1 = np.array(n0)
    n0[2]= 0.
    for item in n1:
        print(item)

@clockTester
def test11():
    n0 = np.array([i + 10 for i in range(20)])
    for i in range(20000):
        a = len(n0)

@clockTester
def test12():
    n0 = np.array([i + 10 for i in range(20)])
    for i in range(20000):
        a = n0.size

@clockTester
def test13():
    a0 = np.array([1., 2., 3. ])
    print(a0)
    for item in a0:
        item = 0.
    print(a0)

def test14():
    a0 = np.array([1., 2., 3.])
    a1 = np.array([1., 1., 1.])
    print(np.var(a0))
    print(np.var(a1))

def test15():
    print(int(np.around(0.9, 0)))

def test16():
    rFitnessList = [1., 2., 3.]
    bestRIndex = 0
    badRIndex = 0
    for i in range(len(rFitnessList)):
        if rFitnessList[i] < rFitnessList[bestRIndex]:
            bestRIndex = i
        if rFitnessList[i] > rFitnessList[badRIndex]:
            badRIndex = i
    if bestRIndex == badRIndex:
        middleRIndex = bestRIndex
    else:
        middleRIndex = ({0, 1, 2} - {bestRIndex, badRIndex}).pop()

    bestR = rFitnessList[bestRIndex]
    badR = rFitnessList[badRIndex]
    middleR = rFitnessList[middleRIndex]
    print('bestR:{0} \r\n badR:{1} \r\n middleR:{2}'.format(bestR, badR, middleR))

def test17():
    l0 = [[[0, 1], [3, 6]], [[-0, -1], [-3, -6]]]
    print(l0[0])

def test18():
    n0 = np.array([[1., .5],
                   [8., 9.],
                   [.9, .89]])
    print(n0)
    print(np.min(n0, axis=0))
    print(np.min(n0, axis=1))
def main():
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    # test8()
    # test9()
    # test10()
    # test12()
    # test11()
    # test13()
    # test14()
    # test15()
    # test16()
    # test17()
    test18()


if __name__ == "__main__":
    main()