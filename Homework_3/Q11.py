import math
import time
import numpy as np
import random as rd
import itertools

def phi(x):
    z = []
    for i in range(len(x)):
        z.append([x[i][0], x[i][1], math.pow(x[i][0], 2), x[i][0] * x[i][1], math.pow(x[i][1], 2)])
    return np.array(z)

def PLA(x, y, arr, eta, outTime):
    w = np.zeros(len(x[0]), dtype=np.float)
    update = 0.0
    t0 = time.time()
    while errRate(x, y, w) != 0:
        for i in arr:
            sign = 1.0 if np.dot(x[i], w) > 0 else -1.0
            if sign != y[i]:
                w = w + eta * y[i] * x[i]
                update += 1.0
        t1 = time.time()
        if t1 - t0 >= outTime:
            return False
    return (w, update)

def errRate(x, y, w):
    yS = np.dot(x, w)
    yS[yS <= 0] = -1.0
    yS[yS > 0] = 1.0
    yErr = yS[yS != y]
    errCount = yErr.shape[0]
    return float(errCount) / len(y)

def main():
    outTime = 60
    TRAIN11_FILE = 'ML_HW3_Q11.dat'

    TRAIN11_DATA = np.loadtxt(TRAIN11_FILE, dtype=np.float)
    x11 = np.column_stack((np.ones(TRAIN11_DATA.shape[0]), phi(TRAIN11_DATA[:, 0:TRAIN11_DATA.shape[1]])))

    t0 = time.time()
    wList = []
    upList = []
    plaResult = True
    for y11 in itertools.product([-1, 1], repeat=TRAIN11_DATA.shape[0]):
        plaResult = PLA(x11, y11, list(xrange(len(x11))), 1.0, outTime)
        if plaResult:
            w11 = plaResult[0]
            update = plaResult[1]
        else:
            break
        wList.append(w11)
        upList.append(update)
    t1 = time.time()
    print '========================================================='
    if plaResult:
        print 'Question 11:', TRAIN11_DATA.shape[0], 'points shattred'
    else:
        print 'Question 11: time out error'
    print '---------------------------------------------------------'
    print 'Q11 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()