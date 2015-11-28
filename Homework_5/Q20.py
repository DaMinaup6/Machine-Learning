import time
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn import svm

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

def main():
    DIGIT = 0
    REPEAT = 100
    RANDOM_SAMPLE = 1000

    TRAIN20_FILE   = 'hw5_train.dat'
    TRAIN20_DATA_X = np.loadtxt(TRAIN20_FILE, dtype=np.float)
    TRAIN20_DATA_Y = np.loadtxt(TRAIN20_FILE, dtype=np.float)
    x20            = TRAIN20_DATA_X[:, 1:TRAIN20_DATA_X.shape[1]]
    y20            = TRAIN20_DATA_Y[:, 0:1]

    eValHist = []
    gamList  = [0, 1, 2, 3, 4]
    t0 = time.time()
    for times in range(REPEAT):
        randIdx  = np.array(rd.sample(range(len(x20)), RANDOM_SAMPLE))
        pRandIdx = np.array([r for r in range(len(x20)) if r not in randIdx])
        xTrain20 = x20[randIdx]
        yTrain20 = y20[randIdx]
        xValid20 = x20[pRandIdx]
        yValid20 = y20[pRandIdx]
        
        yTrain20[yTrain20 != DIGIT] = -1.0
        yTrain20[yTrain20 == DIGIT] = 1.0
        yArr = []
        for y in yTrain20:
            yArr.append(y[0])
        yTrain20 = np.array(yArr)

        yValid20[yValid20 != DIGIT] = -1.0
        yValid20[yValid20 == DIGIT] = 1.0
        yArr = []
        for y in yValid20:
            yArr.append(y[0])
        yValid20 = np.array(yArr)

        eValList = []
        for gamPower in range(0, 5):
            clf  = svm.SVC(C=0.1, kernel='rbf', gamma=(10 ** gamPower))
            clf.fit(xTrain20, yTrain20)
            eVal = 1 - clf.fit(xTrain20, yTrain20).score(xValid20, yValid20)
            eValList.append(eVal)
        print eValList
        eValHist.append(np.argmin(eValList))
        print eValHist
    plotHist(eValHist, r"$\log_{10}\gamma$", r'$E_{\mathrm{val}}$', "Q20", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 20:', eValHist
    print '---------------------------------------------------------'
    print 'Q20 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()