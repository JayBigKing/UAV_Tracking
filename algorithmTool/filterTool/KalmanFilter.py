import numpy as np
from copy import deepcopy

DEFAULT_X_2D_MAT = np.mat([[0.,],[0.,]])
DEFAULT_P_2D_MAT = np.mat([[1.,0.],[1.,0.]])
DEFAULT_A_2D_MAT = np.mat([[1.,1.],[0.,1.]])
DEFAULT_H_2D_MAT = np.mat([[1., 0.], [0., 1.]])
DEFAULT_Q_2D_MAT = np.mat([[0.001, 0.], [0., 0.001]])
DEFAULT_R_2D_MAT = np.mat([[10., 0.], [0., 10.]])

DEFAULT_X_1D_MAT = np.mat([[0.,]])
DEFAULT_P_1D_MAT = np.mat([[0.]])
DEFAULT_A_1D_MAT = np.mat([[1.]])
DEFAULT_H_1D_MAT = np.mat([[1.]])
DEFAULT_Q_1D_MAT = np.mat([[1]])
DEFAULT_R_1D_MAT = np.mat([100.])

class KalmanFilter:
    def __init__(self):
        pass
    def __clear(self):
        self.__X = []
        self.__P = []
        self.__A = []
        self.__H = []
        self.__Q = []
        self.__R = []
        self.__Z = []
        self.__K = None
    def __initCommon(self,xMat ,pMat ,
             hMat ,qMat ,rMat ,):
        self.__clear()

        if isinstance(xMat,np.matrix):
            self.__X = xMat
        else:
            self.__X = np.mat(xMat)

        if isinstance(qMat,np.matrix):
            self.__Q = qMat
        else:
            self.__Q = np.mat(qMat)

        if isinstance(rMat,np.matrix):
            self.__R = rMat
        else:
            self.__R = np.mat(rMat)
        if isinstance(pMat,np.matrix):
            self.__P = pMat
        else:
            self.__P = np.mat(pMat)

        if isinstance(hMat,np.matrix):
            self.__H = hMat
        else:
            self.__H = np.mat(hMat)

    def init2D(self,xMat = DEFAULT_X_2D_MAT,pMat = DEFAULT_P_2D_MAT,aMat = DEFAULT_A_2D_MAT,
             hMat = DEFAULT_H_2D_MAT,qMat = DEFAULT_Q_2D_MAT,rMat = DEFAULT_R_2D_MAT,timeInval = None):
        '''2阶卡尔曼滤波初始化
        input:
        output:
        '''
        self.__initCommon(xMat,pMat,hMat,qMat,rMat)
        if timeInval is not None:
            self.__A = np.mat([[1,timeInval],[0,1]])
        elif isinstance(aMat,np.matrix):
            self.__A = aMat
        else:
            self.__A = np.mat(aMat)
        self.__Z = np.mat([[0,],[0,]])

    def init1D(self, xMat=DEFAULT_X_1D_MAT, pMat=DEFAULT_P_1D_MAT, aMat=DEFAULT_A_1D_MAT,
               hMat=DEFAULT_H_1D_MAT, qMat=DEFAULT_Q_1D_MAT, rMat=DEFAULT_R_1D_MAT, timeInval=None):
        self.__initCommon(xMat,pMat,hMat,qMat,rMat)
        if timeInval is not None:
            self.__A = np.mat([[timeInval]])
        elif isinstance(aMat,np.matrix):
            self.__A = aMat
        else:
            self.__A = np.mat(aMat)

        self.__Z = np.mat([[0., ]])


    def __kalManCommon(self,dim):
        #1. X'(k) = A * X(k - 1)
        xMinus = self.__A * self.__X

        #2. P'(k) = A * P(k-1) * A^(t) + Q
        pMinus = self.__A * self.__P * self.__A.T + self.__Q

        #3 Kalman(k) = P'(k) * H^(t) / (H * P'(k) * H^(t) + R)
        # self.__K = pMinus * self.__H.T / (self.__H * pMinus * self.__H.T + self.__R)
        self.__K = np.linalg.inv((self.__H * pMinus * self.__H.T + self.__R))
        self.__K = pMinus * self.__H.T * self.__K


        #4 X(k) = X'(k) + Kalman(k) * (Z(k) - H * X'(k))
        self.__X = xMinus + self.__K * (self.__Z - self.__H * xMinus)

        #5 P(k) = (1 - Kalman(k) * H ) P'(k)
        #np.eye(2) 是 2*2 单位阵
        self.__P = (np.eye(dim) - self.__K * self.__H) * pMinus



    def kalmanFilterCalc1D(self,signal):
        self.__Z[0,0] = signal
        self.__kalManCommon(1)
        return self.__X[0,0]

    def kalmanFilterCalc2D(self,signal1,signal2):
        self.__Z[0,0] = signal1
        self.__Z[1,0] = signal2
        self.__kalManCommon(2)
        return self.__X[0,0],self.__X[1,0]

    def kalmanFilterCalc2DMulti(self,signal1,signal2, multiTimes):
        self.__Z[0, 0] = signal1
        self.__Z[1, 0] = signal2
        self.__kalManCommon(2)
        res = [[self.__X[0,0],self.__X[1,0]]]

        orginK = deepcopy(self.__K)
        orginX = deepcopy(self.__X)
        orginP = deepcopy(self.__P)
        for i in range(multiTimes - 1):
            xMinus = self.__A * self.__X
            self.__Z[0,0] = xMinus[0, 0]
            self.__Z[1,0] = xMinus[1, 0]
            self.__kalManCommon(2)
            res.append([self.__X[0,0],self.__X[1,0]])

        self.__K = orginK
        self.__X = orginX
        self.__P = orginP

        return res

# -*- coding: utf-8 -*-
import numpy as np

class KalmanFilter2(object):
    def __init__(self,F = np.eye(4),H = np.eye(4),R_val = 10. ,Q_val = 0.001):
        self.F =F # 预测时的矩阵
        self.H = H # 测量时的矩阵
        self.n=self.F.shape[0]
        self.Q = np.zeros((self.n,self.n))
        self.Q[2,2]=Q_val
        self.Q[3,3]=Q_val
        self.R = np.zeros((4,4))
        self.R[0,0]=R_val
        self.R[1,1]=R_val
        self.R[2,2]=R_val
        self.R[3,3]=R_val
        self.P = np.eye(self.n)
        self.B = np.zeros((self.n, 1))
        self.state=0

    #第一次传入时设置观测值为初始估计值
    def set_state(self,x,y,time_stamp):
        self.X = np.zeros((self.n, 1))
        self.speed_x=0
        self.speed_y=0
        self.X=np.array([[x],[y],[self.speed_x],[self.speed_y]])
        self.pre_X=self.X
        self.time_stamp=time_stamp
        self.duration=0

    def process(self,x,y,time_stamp = -1., deltaTime = -1.):
        if self.state==0:
            self.set_state(x,y,time_stamp)
            self.state=1
            return x,y
        if deltaTime < 0:
            self.duration=(time_stamp-self.time_stamp).seconds
            self.time_stamp=time_stamp
        else:
            self.duration = deltaTime
        self.Z=np.array([[x],[y],[self.speed_x],[self.speed_y]])
        #更新时长
        self.F[0,2]=self.duration
        self.F[1,3]=self.duration
        self.predict()
        self.update()
        return self.X[0,0],self.X[1,0]

    def multiProcess(self,n, x,y,time_stamp = -1., deltaTime = -1.):
        firstStateFlag = False
        if n == 1:
            return self.process(x, y, deltaTime=deltaTime)
        Xlist = np.zeros((n,2))
        if self.state==0:
            self.set_state(x,y,time_stamp)
            self.state=1
            firstStateFlag = True
            Xlist[0,0], Xlist[0,1] = x, y
        else:
            Xlist[0,0], Xlist[0,1] = self.process(x,y,deltaTime=deltaTime)
        if firstStateFlag is False:
            originZ = deepcopy(self.Z)
        originF = deepcopy(self.F)
        originX = deepcopy(self.X)
        originP = deepcopy(self.P)
        for i in range(1, n):
            Xlist[i, 0], Xlist[i, 1] = self.process(Xlist[i - 1, 0], Xlist[i - 1, 1], deltaTime=deltaTime)
        if firstStateFlag is False:
            self.Z = originZ
        self.F = originF
        self.X = originX
        self.P = originP

        return Xlist





    # 预测
    def predict(self, u = 0):
        # 实现公式x(k|k-1)=F(k-1)x(k-1)+B(k-1)u(k-1)
        # np.dot(F,x)用于实现矩阵乘法
        self.X = np.dot(self.F, self.X) + np.dot(self.B, u)
        # 实现公式P(k|k-1)=F(k-1)P(k-1)F(k-1)^T+Q(k-1)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q


    # 状态更新,使用观测校正预测
    def update(self):
        # 新息公式y(k)=z(k)-H(k)x(k|k-1)
        y = self.Z - np.dot(self.H, self.X)
        # 新息的协方差S(k)=H(k)P(k|k-1)H(k)^T+R(k)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        # 卡尔曼增益K=P(k|k-1)H(k)^TS(k)^-1
        # linalg.inv(S)用于求S矩阵的逆
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # 状态更新，实现x(k)=x(k|k-1)+Ky(k)
        self.X = self.X + np.dot(K, y)
        #计算速度
        self.speed_x=(self.X[0,0]-self.pre_X[0,0])/self.duration
        self.speed_y=(self.X[1,0]-self.pre_X[1,0])/self.duration
        self.pre_X=self.X
        # 定义单位阵
        I = np.eye(self.n)
        # 估计值vs真实值 协方差更新
        # P(k)=[I-KH(k)]P(k|k-1)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
