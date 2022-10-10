import logging
import colorlog
import time
from numba import cuda
import numpy as np
import math


def testLog():
    log_colors_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    loggerFileName = r'./logs/%s%s.log' % ("log", str(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())))

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # logging.basicConfig(level=logging.INFO #设置日志输出格式
    #                     ,filename=loggerFileName #log日志输出的文件位置和文件名
    #                     ,filemode="w" #文件的写入格式，w为重新写入文件，默认是追加
    #                     ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
    #                     # -8表示占位符，让输出左对齐，输出长度都为8位
    #                     ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
    #                     )
    # logging.info("info")
    # logging.debug("debug")
    # logging.warning("warn")
    # logging.error("error")

    logger = logging.getLogger("logger1")
    sh1 = logging.StreamHandler()
    fh1 = logging.FileHandler(filename="fh.log", mode='w')

    # 格式器
    fmt1 = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s : %(lineno)s line - %(message)s")
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=log_colors_config
    )

    sh1.setFormatter(console_formatter)
    fh1.setFormatter(console_formatter)
    sh1.setLevel(logging.DEBUG)
    fh1.setLevel(logging.DEBUG)

    logger.addHandler(fh1)
    logger.addHandler(sh1)

    # logger.setLevel(logging.DEBUG)
    logger.info("info")
    logger.debug("debug")
    logger.warning("warn")
    logger.error("error")



@cuda.jit
def gpu_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n :
        result[idx] = a[idx] + b[idx]

@cuda.jit
def gpu_add2(a, b, result, n, dim):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n :
        for i in range(dim):
            result[idx, i] = a[idx, i] + b[idx, i]

def testCuda():
    from time import time
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    # 拷贝数据到设备端
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    # 在显卡设备上初始化一块用于存放GPU计算结果的空间
    gpu_result = cuda.device_array(n)
    cpu_result = np.empty(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x_device, y_device, gpu_result, n)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    cpu_result = np.add(x, y)
    print("cpu vector add time " + str(time() - start))

    if (np.array_equal(cpu_result, gpu_result.copy_to_host())):
        print("result correct!")

def testCuda2():
    from time import time
    n = 50
    x = np.ones((25, 25))
    y = 2 * x

    # 拷贝数据到设备端
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    # 在显卡设备上初始化一块用于存放GPU计算结果的空间
    gpu_result = cuda.device_array((25, 25))
    cpu_result = np.empty((25, 25))

    threads_per_block = 5
    blocks_per_grid = math.ceil(25 / threads_per_block)
    start = time()
    gpu_add2[blocks_per_grid, threads_per_block](x_device, y_device, gpu_result, 25, 25)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    cpu_result = np.add(x, y)
    print("cpu vector add time " + str(time() - start))

    if (np.array_equal(cpu_result, gpu_result.copy_to_host())):
        print("result correct!")
    print(gpu_result.copy_to_host())

testCuda2()