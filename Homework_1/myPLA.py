import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt

def PLA(x, y, arr, eta):
    w = np.zeros(len(x[0]), dtype=np.float)
    update = iLast = 0.0
    while errRate(x, y, w) != 0:
        for i in arr:
            sign = 1.0 if np.dot(x[i], w) > 0 else -1.0
            if sign != y[i]:
                w = w + eta * y[i] * x[i]
                update += 1.0
                iLast = i
    return (w, update, iLast)

def pocketAlgorithm(x, y, arr, eta, haltUp):
    oW = pW = np.zeros(len(x[0]), dtype=np.float)
    update = pocErr = oriErr = 0.0
    while update < haltUp:
        for i in arr:
            sign = 1.0 if np.dot(x[i], oW) > 0 else -1.0
            if sign != y[i]:
                oW = oW + eta * y[i] * x[i]
                oriErr = errRate(x, y, oW)
                if pocErr == 0: pocErr = errRate(x, y, pW)
                update += 1.0
                if pocErr > oriErr:
                    pW = oW
                    pocErr = oriErr
                if update == haltUp: break
    return (oW, pW)

def randomSeedPLA(x, y, eta, ranTimes):
    sumUp = 0.0
    arr = list(range(len(x)))
    updateList = []
    totalTime = 0.0
    for times in range(ranTimes):
        rd.shuffle(arr)
        t0 = time.time()
        update = PLA(x, y, arr, eta)[1]
        t1 = time.time()
        updateList.append(update)
        sumUp += update
        totalTime += (t1 - t0)
    return (sumUp / ranTimes, np.array(updateList, dtype=int), totalTime / ranTimes)

def randomSeedPoc(xTrain, yTrain, xTest, yTest, eta, haltUp, ranTimes, usePocket):
    sumPocNum = 0.0
    arr = list(range(len(xTrain)))
    errList = []
    for times in range(ranTimes):
        rd.shuffle(arr)
        (originW, pocketW) = pocketAlgorithm(xTrain, yTrain, arr, eta, haltUp)
        if usePocket:
            err = errRate(xTest, yTest, pocketW)
            sumPocNum += err
            errList.append(err)
        else:
            err = errRate(xTest, yTest, originW)
            sumPocNum += err
            errList.append(err)
    return (sumPocNum / ranTimes, np.array(errList))

def errRate(x, y, w):
    yS = np.dot(x, w)
    yS[yS <= 0] = -1.0
    yS[yS > 0] = 1.0
    yErr = yS[yS != y]
    return float(yErr.shape[0]) / len(y)

def PLATime(x, y, eta, repeats, ranTimes, scalDown0, scalDown1):
    totalFactor = 0.0
    factorList = []
    for times in range(repeats):
        time0 = randomSeedPLA((x / scalDown0), y, eta, ranTimes)[2]
        time1 = randomSeedPLA((x / scalDown1), y, eta, ranTimes)[2]
        factor = time1 / time0
        totalFactor += factor
        factorList.append(factor)
    return (totalFactor / repeats, np.array(factorList))

def plotHist(x, xLabel, yLabel, title, width, isFloat):
    plt.title(str(title))
    plt.xlabel(str(xLabel))
    plt.ylabel(str(yLabel))
    if isFloat: plt.hist(x)
    else:
        freq = np.bincount(x)
        freqIndex = np.nonzero(freq)[0]
        plt.bar(freqIndex, freq[freqIndex], width)
    plt.grid(True)
    plt.draw()
    plt.savefig(title)
    plt.close()