#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : testMultiProcess.py
@Author  : jay.zhu
@Time    : 2022/12/17 22:04
"""
import numpy as np
import time
from multiprocessing import Process, Pipe, Manager
from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester

processNum = 4
baseVal = 10000
list0 = [[float(j) for j in range(baseVal * i, baseVal * (i + 1))] for i in range(processNum)]
resList = [0. for i in range(processNum)]


def processWork(index):
    totalVal = 0.
    for val in list0[index]:
        totalVal += val
    resList[index] = totalVal


@clockTester
def test1():
    for i in range(processNum):
        processWork(i)
    print(r'total num is %f' % sum(resList))


@clockTester
def test2():
    process_list = [Process(target=processWork, args=(i,)) for i in range(processNum)]
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    print(r'total num is %f' % sum(resList))


def test3_testAdd(pipe):
    counter = 1
    while counter < 10:
        pipe.send({
            "val": 'testAdd : %d' % counter,
            "terminal": False,
        })
        counter += 1

    pipe.send({
        "val": '',
        "terminal": True,
    })


def test3_testMul(pipe):
    muler = 1.
    while muler < 32.:
        pipe.send({
            "val": 'testMul : %d' % muler,
            "terminal": False,
        })
        muler *= 2.

    pipe.send({
        "val": '',
        "terminal": True,
    })


def test3_testMana():
    funcList = [test3_testAdd]
    pipeList = [Pipe() for item in funcList]
    processList = [Process(target=item, name=item.__name__, args=(pipeList[index][0],)) for index, item in
                   enumerate(funcList)]
    for item in processList:
        item.start()
        item.join()

    terminalCount = 0
    while True:
        if terminalCount >= len(funcList):
            break
        for item in pipeList:
            ret = item[1].recv()

            if ret["terminal"] is True:
                terminalCount += 1
            else:
                print(ret["val"])


@clockTester
def test3():
    # test3_testMana()
    process = Process(target=test3_testMana, name=test3_testMana.__name__, args=())
    process.start()
    process.join()


class Test4Base:
    def __init__(self):
        self.valName = None
        self.terminalName = None
        self.val = [1.]
        self.terminal = [False]

    def managerProcessConn(self, managerDict, managerList):
        if self.valName is not None:
            managerDict[self.valName] = self.val
        if self.terminalName is not None:
            managerDict[self.terminalName] = self.terminal

    def process(self, managerDict, managerList):
        self.managerProcessConn(managerDict, managerList)

    def getName(self):
        return self.__class__.__name__



class Test4Add(Test4Base):
    def __init__(self):
        super().__init__()
        self.valName = "val" + self.__class__.__name__
        self.terminalName = "terminal" + self.__class__.__name__

    @clockTester
    def process(self, managerDict, managerList):
        super().process(managerDict, managerList)
        counter = self.val[0]
        while counter < 50:
            counter += 1
            managerDict[self.valName] = [counter]
            time.sleep(0.5)
        managerDict[self.terminalName] = [True]


    def getName(self):
        return self.__class__.__name__


class Test4Mul(Test4Base):
    def __init__(self):
        super().__init__()
        self.valName = "val" + self.__class__.__name__
        self.terminalName = "terminal" + self.__class__.__name__

    @clockTester
    def process(self, managerDict, managerList):
        super().process(managerDict, managerList)
        muler = self.val[0]
        while muler < 512:
            muler *= 2
            managerDict[self.valName] = [muler]
            time.sleep(0.5)
        managerDict[self.terminalName] = [True]

    def getName(self):
        return self.__class__.__name__


def test4_mana():
    with Manager() as manager:
        classList = [Test4Add, Test4Mul]
        dictList = [manager.dict() for item in classList]
        listList = [manager.list(range(10)) for item in classList]
        objList = [item() for item in classList]
        objDictKeyList = [(item.valName, item.terminalName) for item in objList]
        processList = [Process(target=item.process, name=item.getName(), args=(dictList[index], listList[index],)) for
                       index, item in
                       enumerate(objList)]
        terminalFlagList = [False for item in classList]

        for item in processList:
            item.start()
            # item.join()

        terminalCount = 0
        while True:
            if terminalCount == len(objList):
                break

            for index, item in enumerate(objDictKeyList):
                if terminalFlagList[index] is False:
                    if dictList[index].get(item[1]) is not None:
                        if dictList[index][item[1]][0] is True:
                            terminalFlagList[index] = True
                            terminalCount += 1
                        else:
                            if dictList[index].get(item[0]) is not None:
                                print("%s :  %f" % (classList[index].__name__, dictList[index][item[0]][0]))



def test4():
    # test4_mana()
    process = Process(target=test4_mana, name=test4_mana.__name__, args=())
    process.start()
    process.join()


def test5():
    with Manager() as manager:
        d = manager.dict()
        l = manager.list(range(10))

        list = [1, 2, 3, 4, 5]
        sett = {1, 2, 3, 4, 5}
        d["1"] = list
        d["2"] = sett
        print(d["1"])
        list[0] = 100
        sett.add(100)
        print(d["1"])
        print(d["2"])


def main():
    # global list0
    # global resList
    # test1()
    #
    # test2()
    # test3()
    test4()


if __name__ == "__main__":
    main()
