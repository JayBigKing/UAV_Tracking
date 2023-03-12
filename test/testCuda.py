#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@Project : UAV_Tracking
@File    : testCuda.py
@Author  : jay.zhu
@Time    : 2023/2/25 12:26
"""
from time import time
import functools
import math
import numpy as np
import random
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


def printTestRes(ifPrint=True):
    def callFunc(func):
        @functools.wraps(func)
        def dataPrinter(*args, **kwargs):
            res = func(*args, **kwargs)
            dataFromGpu = res.copy_to_host()
            if ifPrint is True:
                print(dataFromGpu)
            return dataFromGpu
        return dataPrinter
    return callFunc



def testBase(n, dim):
    x = np.ones((dim, n))
    y = np.array(x)
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    gpu_result = cuda.device_array((dim, n))
    return x, y, x_device, y_device, gpu_result

@cuda.jit
def gpu_func1Pos(result, n, dim):
    i, j = cuda.grid(2)
    if i < dim and j < n:
        result[i, j] = i + j

@printTestRes()
def testCuda1():
    n = 10
    dim = 25
    x, y, x_device, y_device, gpu_result = testBase(n, dim)

    threads_per_block = [5, 5]
    blocks_per_grid = [math.ceil(dim / threads_per_block[0]), math.ceil(n / threads_per_block[0])]
    start = time()
    gpu_func1Pos[blocks_per_grid, threads_per_block](gpu_result, 10, 25)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    # start = time()
    # cpu_result = np.add(x, y)
    # print("cpu vector add time " + str(time() - start))

    return gpu_result

@cuda.jit
def gpu_func2(a, result, n, dim):
    idx = cuda.grid(1)
    if idx == 0:
        result[0] = 0.
        for val in a[idx, :]:
            result[0] += val
        for i in a.shape[0]:
            result[0] /= 5.

@printTestRes()
def testCuda2():
    n = 10
    dim = 1
    x, y, x_device, y_device, gpu_result = testBase(n, dim)
    gpu_result = cuda.device_array(1)

    start = time()
    gpu_func2[1, 1](x_device, gpu_result, 10, 25)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    cpu_result = np.sum(x, axis=1)
    print("cpu vector add time " + str(time() - start))

    return gpu_result

@cuda.jit(device=True)
def gpu_func3Help2(a):
    return a * 1.7

@cuda.jit(device=True)
def gpu_func3Help(a, b):
    return a*2, gpu_func3Help2(b)

@cuda.jit
def gpu_func3(a, b, result, n, dim):
    i, j = cuda.grid(2)
    if i < dim and j < n:
        aVal, bVal = gpu_func3Help(a[i, j], b[i, j])
        result[i, j]= aVal, bVal

@printTestRes()
def testCuda3():
    n = 10
    dim = 20
    x, y, x_device, y_device, gpu_result = testBase(n, dim)

    threads_per_block = [5, 5]
    blocks_per_grid = [math.ceil(dim / threads_per_block[0]), math.ceil(n / threads_per_block[0])]
    start = time()
    gpu_func3[threads_per_block, blocks_per_grid](x_device,y_device, gpu_result, n, dim)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))

    return gpu_result

@cuda.jit(device=True)
def gpu_randint(rng_states, threadIdx, a, b):
    randRes = xoroshiro128p_uniform_float32(rng_states, threadIdx)
    return int(a + randRes * (b - a + 1))
    # randRes = xoroshiro128p_uniform_float32(rng_states, threadIdx)
    # numOfCouldSelect = b - a + 1
    # eachIntProb = 1. / numOfCouldSelect
    # left = 0
    # right = numOfCouldSelect - 1
    # selectIntIndex = (left + right) / 2

    # while True:
    #     if selectIntIndex == 0:
    #         if randRes < eachIntProb:
    #             return a + selectIntIndex
    #     elif selectIntIndex == numOfCouldSelect - 1:
    #         if randRes > eachIntProb * selectIntIndex:
    #             return a + selectIntIndex
    #     else:
    #         if randRes < eachIntProb * selectIntIndex:
    #             right = selectIntIndex - 1
    #         elif randRes > eachIntProb * (selectIntIndex + 1):
    #             left = selectIntIndex + 1
    #         else:
    #             return a + selectIntIndex
    #
    #         selectIntIndex = int((left + right) / 2)

@cuda.jit
def gpu_func4(result, n, dim, rng_states):
    i, j = cuda.grid(2)
    if i < dim and j < n:
        # result[i, j] = gpu_randint(rng_states, i * cuda.gridDim.x + j, 1., 10.)
        result[i, j]= xoroshiro128p_uniform_float32(rng_states, 1)

@printTestRes()
def testCuda4():
    n = 10
    dim = 20
    x, y, x_device, y_device, gpu_result = testBase(n, dim)

    rng_states = create_xoroshiro128p_states(n * dim, seed=int(time() * 1e7))
    threads_per_block = [5, 5]
    blocks_per_grid = [math.ceil(dim / threads_per_block[0]), math.ceil(n / threads_per_block[0])]
    start = time()
    gpu_func4[threads_per_block, blocks_per_grid](gpu_result, n, dim, rng_states)
    cuda.synchronize()
    print(gpu_result.copy_to_host())
    print("gpu vector add time " + str(time() - start))
    gpu_func4[threads_per_block, blocks_per_grid](gpu_result, n, dim, rng_states)
    cuda.synchronize()
    print(gpu_result.copy_to_host())

    return gpu_result

@cuda.jit
def gpu_func5(a, result, n):
    idx = cuda.grid(1)
    if idx < n:
        a[0, idx], result[5, idx] = 9, 6
        # for i in range(result.shape[0]):
        #     result[i, idx] = idx + 1
    else:
        return

@printTestRes()
def testCuda5():
    n = 10
    dim = 20
    x, y, x_device, y_device, gpu_result = testBase(n, dim)

    threads_per_block = 5
    blocks_per_grid = math.ceil(n / threads_per_block)
    start = time()
    gpu_func5[threads_per_block, blocks_per_grid](x_device, gpu_result, n)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))

    print(x)
    print(x_device.copy_to_host())
    nowTime = time()
    print(nowTime)
    print(int(nowTime * 1e7))
    return gpu_result

@cuda.jit
def gpu_func6(a, result, n):
    idx = cuda.grid(1)
    if idx < n:
        for i in range(result.shape[0]):
            result[i, idx] = idx + 1
    else:
        return

@printTestRes()
def testCuda6():
    class CudaTester():
        def __init__(self):
            self.n = 40
            self.dim = 20
            self.x, self.y, self.x_device, self.y_device, self.gpu_result = testBase(self.n, self.dim)

        def run(self):
            threads_per_block = 5
            blocks_per_grid = math.ceil(self.n / threads_per_block)
            start = time()
            gpu_func6[threads_per_block, blocks_per_grid](self.x_device, self.gpu_result, self.n)
            cuda.synchronize()
            print("gpu vector add time " + str(time() - start))
            return self.gpu_result

    ct = CudaTester()
    return ct.run()

def main():
    testCuda6()

if __name__ == "__main__":

    main()