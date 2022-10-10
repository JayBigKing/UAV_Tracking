import numpy as np
from KalmanFilter import KalmanFilter
import sys
sys.path.append("../../visualize")

from CoorDiagram import CoorDiagram

X_raw = [i for i in range(500)]
V_raw = np.ones(500)
# 创建一个均值为0，方差为1的高斯噪声，共有500个samples，精确到小数点后两位
noiseX = np.round(np.random.normal(0, 1, 500), 2)
noiseV = np.round(np.random.normal(0, 0.18, 500), 2)
# 将z的观测值和噪声相加
X = np.mat(X_raw) + np.mat(noiseX)
V = V_raw + np.mat(noiseV)


kf = KalmanFilter()
cd = CoorDiagram()
kf.init2D(xMat=np.mat([[X[0,0],],[1,]]),timeInval=1)
calcNum = int(100)
# theXList = [X_raw[:calcNum],X_raw[:calcNum]]
# theYList = []
predictXList = []
predictVList = []

for i in range(calcNum):
    predictX , predictV = kf.kalmanFilterCalc2D(X[0,i],V[0,i])
    predictXList.append([float(X_raw[i]), predictX])
    predictVList.append([float(X_raw[i]), predictV])

scattersList = [predictXList,predictVList]
# theYList.append(predictXList)
# theYList.append(predictVList)
cd.drwaManyScattersInOnePlane(scattersList)
cd.drwaManyScatters(scattersList)
cd.drawOneScatter(scattersList[1])
