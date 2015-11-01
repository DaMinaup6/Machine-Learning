import numpy as np
import random as rd
import time

def PLA(x, y, arr, eta):
    w = np.zeros(len(x[0]), dtype=np.float)
    update = 0.0
    iLast = 0.0
    while errRate(x, y, w) != 0:
        for i in arr:
            sign = 1.0 if np.dot(x[i], w) > 0 else -1.0
            if sign != y[i]:
                w = w + eta * y[i] * x[i]
                update += 1.0
                iLast = i
    return (w, update, iLast)

def errRate(x, y, w):
    yS = np.dot(x, w)
    yS[yS <= 0] = -1.0
    yS[yS > 0] = 1.0
    yErr = yS[yS != y]
    errCount = yErr.shape[0]
    return float(errCount) / len(y)

def main():
    TRAIN15_FILE = "ntumlone%2Fhw1%2Fhw1_15_train.dat"

    TRAIN15_DATA = np.loadtxt(TRAIN15_FILE, dtype=np.float)
    x15 = np.column_stack((np.ones(TRAIN15_DATA.shape[0]), TRAIN15_DATA[:, 0:(TRAIN15_DATA.shape[1] - 1)]))
    y15 = TRAIN15_DATA[:, (TRAIN15_DATA.shape[1] - 1)]

    print "========================================================="
    t0 = time.time()
    (w15, update, index) = PLA(x15, y15, list(xrange(len(x15))), 1.0)
    t1 = time.time()
    print "Question 15 number of updates: ", int(update), ", last index: ", int(index)
    print "---------------------------------------------------------"
    print "Q15 costs ", t1 - t0, ' seconds'
    print "========================================================="

if __name__ == '__main__':
    main()