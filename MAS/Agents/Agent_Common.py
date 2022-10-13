import numpy as np
def calcMovingForUAV(x, u, time):
    res = np.zeros(3)
    res[0] = x[0] + u[0] * np.cos(np.deg2rad(x[2])) * time
    res[1] = x[1] + u[0] * np.sin(np.deg2rad(x[2])) * time
    res[2] = x[2] + u[1] * time
    return res

def calcDistance(x0, x1):
    return np.sqrt(np.sum(np.square(x0 - x1)))

def clacDirection(x0, x1):
    return np.rad2deg(np.arctan2(x0[1] - x1[1], x0[0] - x1[0]))