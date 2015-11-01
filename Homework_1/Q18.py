import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt

def pocketAlgorithm(x, y, arr, eta, haltUp):
    oW = pW = np.zeros(len(x[0]), dtype=np.float)
    update = pocErr = oriErr = 0.0
    while update < haltUp:
        for i in arr:
            sign = 1.0 if np.dot(x[i], oW) > 0 else -1.0
            if sign != y[i]:
                oW = oW + eta * y[i] * x[i]
                oriErr = errRate(x, y, oW)
                if pocErr == 0:
                    pocErr = errRate(x, y, pW)
                update += 1.0
                if pocErr > oriErr:
                    pW = oW
                    pocErr = oriErr
                if update == haltUp:
                    break
    return (oW, pW)

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
    errCount = yErr.shape[0]
    return float(errCount) / len(y)

def plotHist(x, xLabel, yLabel, title, width, isFloat):
    plt.title(str(title))
    plt.xlabel(str(xLabel))
    plt.ylabel(str(yLabel))
    if isFloat:
        plt.hist(x)
    else:
        freq = np.bincount(x)
        freqIndex = np.nonzero(freq)[0]
        plt.bar(freqIndex, freq[freqIndex], width)
    plt.grid(True)
    plt.draw()
    plt.savefig(title)
    plt.close()

def main():
    TRAIN18_FILE = "ntumlone%2Fhw1%2Fhw1_18_train.dat"
    TEST18_FILE = "ntumlone%2Fhw1%2Fhw1_18_test.dat"

    TRAIN18_DATA = np.loadtxt(TRAIN18_FILE, dtype=np.float)
    x18 = np.column_stack((np.ones(TRAIN18_DATA.shape[0]), TRAIN18_DATA[:, 0:4]))
    y18 = TRAIN18_DATA[:, 4]

    TEST18_DATA = np.loadtxt(TEST18_FILE, dtype=np.float)
    a = np.column_stack((np.ones(TEST18_DATA.shape[0]), TEST18_DATA[:, 0:4]))
    b = TEST18_DATA[:, 4]

    print "====================================="
    t6 = time.time()
    (aveErr18, errHistQ18) = randomSeedPoc(x18, y18, a, b, 1.0, 50, 2000, True)
    plotHist(errHistQ18, "Error Rate", "Frequency", "Q18", 0.01, True)
    t7 = time.time()
    print "Question 18: ", aveErr18
    print "-------------------------------------"
    print "Q18 costs ", t7 - t6, ' seconds'
    print "====================================="

if __name__ == '__main__':
    main()