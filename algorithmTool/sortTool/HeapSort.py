#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : HeapSort.py
@Author  : jay.zhu
@Time    : 2023/3/10 16:34
"""
from copy import deepcopy

class HeapSort:
    def __init__(self):
        self.cmpFunc = self.defaultCmpFunc
        self.__BIG_LABEL = 2
        self.__SMALL_LABEL = 3
        self.__TOP_LABEL = 4
        self.__BOTTOM_LABEL = 5


    def defaultCmpFunc(self, a, b):
        return (a- b)

    def sort(self, dataList, sortWay=0, cmp=None):
        if cmp is not None:
            self.cmpFunc = cmp
        if len(dataList) == 1:
            return dataList
        else:
            resList = deepcopy(dataList)
            if sortWay == 0:
                self.__bigHeapSort(resList)
            else:
                self.__smallHeapSort(resList)
            return resList

    def checkInputN(self, dataList, n):
        if n <= 0:
            return -1
        elif n > len(dataList):
            return 1
        else:
            return 0

    def __getTopOrBottomNInSortListInner(self, dataList, n, topOrBottom, sortWay=0, cmp=None):
        if cmp is not None:
            self.cmpFunc = cmp
        checkRes = self.checkInputN(dataList, n)
        if checkRes == -1:
            raise ValueError("input should not be lower than or equal to 0")
        elif checkRes == 1:
            raise ValueError("input should not be greater than data list len")
        else:
            if len(dataList) == 1:
                return [dataList[0]]
            else:
                resList = deepcopy(dataList)
                if topOrBottom == self.__TOP_LABEL:
                    if sortWay == 0:
                        self.__smallHeapSort(resList, sortTopN=n)
                    else:
                        self.__bigHeapSort(resList, sortTopN=n)
                else:
                    if sortWay == 0:
                        self.__bigHeapSort(resList, sortTopN=n)
                    else:
                        self.__smallHeapSort(resList, sortTopN=n)
                return resList[len(resList)-1: len(resList)-n-1: -1]

    def getTopNInSortList(self, dataList, n, sortWay=0, cmp=None):
        return self.__getTopOrBottomNInSortListInner(dataList, n, self.__TOP_LABEL, sortWay, cmp)

    def getBottomNInSortList(self, dataList, n, sortWay=0, cmp=None):
        return self.__getTopOrBottomNInSortListInner(dataList, n, self.__BOTTOM_LABEL, sortWay, cmp)


    def __getTopOrBottomNInSortIndexListInner(self, dataList, n, topOrBottom, sortWay=0, cmp=None):
        if cmp is not None:
            self.cmpFunc = cmp
        checkRes = self.checkInputN(dataList, n)
        if checkRes == -1:
            raise ValueError("input should not be lower than or equal to 0")
        elif checkRes == 1:
            raise ValueError("input should not be greater than data list len")
        else:
            if len(dataList) == 1:
                return [dataList[0]]
            else:
                resList = deepcopy(dataList)
                if topOrBottom == self.__TOP_LABEL:
                    if sortWay == 0:
                        indexList = self.__smallHeapSortWithIndex(resList, sortTopN=n)
                    else:
                        indexList = self.__bigHeapSortWithIndex(resList, sortTopN=n)
                else:
                    if sortWay == 0:
                        indexList = self.__bigHeapSortWithIndex(resList, sortTopN=n)
                    else:
                        indexList = self.__smallHeapSortWithIndex(resList, sortTopN=n)
                return indexList[len(resList)-1: len(resList)-n-1: -1]

    def getTopNInSortIndexList(self, dataList, n, sortWay=0, cmp=None):
        return self.__getTopOrBottomNInSortIndexListInner(dataList, n, self.__TOP_LABEL, sortWay, cmp)

    def getBottomNInSortIndexList(self, dataList, n, sortWay=0, cmp=None):
        return self.__getTopOrBottomNInSortIndexListInner(dataList, n, self.__BOTTOM_LABEL, sortWay, cmp)


    def __heapSwapInner(self, dataList, fatherIndex, limitIndex, whichHeap):
        leftChildIndex = int(fatherIndex * 2) + 1
        rightChildIndex = leftChildIndex + 1
        fatherNewIndex = fatherIndex

        if leftChildIndex > limitIndex:
            return
        else:
            if whichHeap == self.__BIG_LABEL:
                if self.cmpFunc(dataList[leftChildIndex], dataList[fatherNewIndex]) > 0:
                    fatherNewIndex = leftChildIndex
                if rightChildIndex <= limitIndex:
                    if self.cmpFunc(dataList[rightChildIndex], dataList[fatherNewIndex]) > 0:
                        fatherNewIndex = rightChildIndex
            else:
                if self.cmpFunc(dataList[leftChildIndex], dataList[fatherNewIndex]) < 0:
                    fatherNewIndex = leftChildIndex
                if rightChildIndex <= limitIndex:
                    if self.cmpFunc(dataList[rightChildIndex], dataList[fatherNewIndex]) < 0:
                        fatherNewIndex = rightChildIndex

        if fatherNewIndex != fatherIndex:
            dataList[fatherIndex], dataList[fatherNewIndex] = dataList[fatherNewIndex], dataList[fatherIndex]
            self.__heapSwapInner(dataList, fatherNewIndex, limitIndex, whichHeap)

    def __bigOrSmallHeapSortInner(self, dataList, bigOrSmall, sortTopN=None):
        if sortTopN is None:
            sortTopNNum = len(dataList) - 1
        else:
            sortTopNNum = sortTopN
        fatherStartIndex = int(len(dataList) / 2) - 1
        limitIndex = len(dataList) - 1
        for count in range(sortTopNNum):
            for i in range(fatherStartIndex, -1, -1):
                self.__heapSwapInner(dataList, i, limitIndex, bigOrSmall)
            dataList[0], dataList[limitIndex] = dataList[limitIndex], dataList[0]
            limitIndex -= 1


    def __bigHeapSort(self, dataList, sortTopN=None):
        self.__bigOrSmallHeapSortInner(dataList, bigOrSmall=self.__BIG_LABEL, sortTopN=sortTopN)

    def __smallHeapSort(self, dataList, sortTopN=None):
        self.__bigOrSmallHeapSortInner(dataList, bigOrSmall=self.__SMALL_LABEL, sortTopN=sortTopN)


    def __heapSwapWithIndexInner(self, dataList, indexList, fatherIndex, limitIndex, whichHeap):
        leftChildIndex = int(fatherIndex * 2) + 1
        rightChildIndex = leftChildIndex + 1
        fatherNewIndex = fatherIndex

        if leftChildIndex > limitIndex:
            return
        else:
            if whichHeap == self.__BIG_LABEL:
                if self.cmpFunc(dataList[leftChildIndex], dataList[fatherNewIndex]) > 0:
                    fatherNewIndex = leftChildIndex
                if rightChildIndex <= limitIndex:
                    if self.cmpFunc(dataList[rightChildIndex], dataList[fatherNewIndex]) > 0:
                        fatherNewIndex = rightChildIndex
            else:
                if self.cmpFunc(dataList[leftChildIndex], dataList[fatherNewIndex]) < 0:
                    fatherNewIndex = leftChildIndex
                if rightChildIndex <= limitIndex:
                    if self.cmpFunc(dataList[rightChildIndex], dataList[fatherNewIndex]) < 0:
                        fatherNewIndex = rightChildIndex

        if fatherNewIndex != fatherIndex:
            dataList[fatherIndex], dataList[fatherNewIndex] = dataList[fatherNewIndex], dataList[fatherIndex]
            indexList[fatherIndex], indexList[fatherNewIndex] = indexList[fatherNewIndex], indexList[fatherIndex]
            self.__heapSwapWithIndexInner(dataList, indexList, fatherNewIndex, limitIndex, whichHeap)

    def __bigOrSmallHeapSortWithIndexInner(self, dataList, bigOrSmall, sortTopN=None):
        if sortTopN is None:
            sortTopNNum = len(dataList) - 1
        else:
            sortTopNNum = sortTopN

        fatherStartIndex = int(len(dataList) / 2) - 1
        limitIndex = len(dataList) - 1
        indexList = [i for i in range(len(dataList))]
        for count in range(sortTopNNum):
            for i in range(fatherStartIndex, -1, -1):
                self.__heapSwapWithIndexInner(dataList, indexList, i, limitIndex, bigOrSmall)
            dataList[0], dataList[limitIndex] = dataList[limitIndex], dataList[0]
            indexList[0], indexList[limitIndex] = indexList[limitIndex], indexList[0]
            limitIndex -= 1
        return indexList


    def __bigHeapSortWithIndex(self, dataList, sortTopN=None):
        return self.__bigOrSmallHeapSortWithIndexInner(dataList, bigOrSmall=self.__BIG_LABEL, sortTopN=sortTopN)

    def __smallHeapSortWithIndex(self, dataList, sortTopN=None):
        return self.__bigOrSmallHeapSortWithIndexInner(dataList, bigOrSmall=self.__SMALL_LABEL, sortTopN=sortTopN)

# hs = HeapSort()
# res = hs.sort(dataList=[2., 3., 4., 0., -1., 9., 2., 7.])
# res2 = hs.sort(dataList=[2., 3., 4., 0., -1., 9., 2., 7.], sortWay=1)
# res3 = hs.getTopNInSortList(dataList=[2., 3., 4., 0., -1., 9., 2., 7.], n=3, sortWay=0)
# res3_index = hs.getTopNInSortIndexList(dataList=[2., 3., 4., 0., -1., 9., 2., 7.], n=3, sortWay=0)
# res4 = hs.getTopNInSortList(dataList=[2., 3., 4., 0., -1., 9., 2., 7.], n=3, sortWay=1)
# res4_index = hs.getTopNInSortIndexList(dataList=[2., 3., 4., 0., -1., 9., 2., 7.], n=3, sortWay=1)
# res5 = hs.getBottomNInSortList(dataList=[2., 3., 4., 0., -1., 9., 2., 7.], n=3, sortWay=0)
# res5_index = hs.getBottomNInSortIndexList(dataList=[2., 3., 4., 0., -1., 9., 2., 7.], n=3, sortWay=0)
# res6 = hs.getBottomNInSortList(dataList=[2., 3., 4., 0., -1., 9., 2., 7.], n=3, sortWay=1)
# res6_index = hs.getBottomNInSortIndexList(dataList=[2., 3., 4., 0., -1., 9., 2., 7.], n=3, sortWay=1)
#
# print(res)
# print(res2)
# print(res3)
# print(res3_index)
# print(res4)
# print(res4_index)
# print(res5)
# print(res5_index)
# print(res6)
# print(res6_index)