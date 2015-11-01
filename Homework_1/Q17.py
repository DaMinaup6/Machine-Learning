import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt

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
    TRAIN15_FILE = "ntumlone%2Fhw1%2Fhw1_15_train.dat"

    TRAIN15_DATA = np.loadtxt(TRAIN15_FILE, dtype=np.float)
    x15 = np.column_stack((np.ones(TRAIN15_DATA.shape[0]), TRAIN15_DATA[:, 0:(TRAIN15_DATA.shape[1] - 1)]))
    y15 = TRAIN15_DATA[:, (TRAIN15_DATA.shape[1] - 1)]

    print "====================================="
    t4 = time.time()
    (aveUpdate17, updateHistQ17, Q17AveTime) = randomSeedPLA(x15, y15, 0.5, 2000)
    plotHist(updateHistQ17, "Updates", "Frequency", "Q17", 1.0, False)
    t5 = time.time()
    print "Question 17: ", aveUpdate17
    print "-------------------------------------"
    print "Q17 costs ", t5 - t4, ' seconds'
    print "====================================="

if __name__ == '__main__':
    main()