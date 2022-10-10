import numpy as np
from Jay_Tool.EfficiencyTestTool import EfficiencyTestTool

@EfficiencyTestTool.clockTester
def test1():
    a = np.zeros((50,100))
    for i in range(50):
        for j in range(100):
            a[i, j] = i * j + np.random.random()

@EfficiencyTestTool.clockTester
def test2():
    a = np.zeros((100,50))
    for j in range(50):
        for i in range(100):
            a[i, j] = i * j + np.random.random()

def main():
    test1()
    test2()

if __name__ == "__main__":
    main()