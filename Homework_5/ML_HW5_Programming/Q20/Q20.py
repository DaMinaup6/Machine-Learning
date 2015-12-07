import time
import numpy as np
import random as rd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn import svm
from joblib import Parallel, delayed

def digitTrans(y, digit):
    y[y != digit] = -1.0
    y[y == digit] = 1.0
    return y

def eValCal():
    DIGIT = 0
    RANDOM_SAMPLE = 1000

    TRAIN20_FILE = 'hw5_train.dat'
    TRAIN20_DATA = np.loadtxt(TRAIN20_FILE, dtype=np.float)
    x20 = TRAIN20_DATA[:, 1:TRAIN20_DATA.shape[1]]
    y20 = digitTrans(TRAIN20_DATA[:, 0], DIGIT)

    gammaList = [0, 1, 2, 3, 4]

    randIdx  = np.array(rd.sample(range(len(x20)), RANDOM_SAMPLE))
    pRandIdx = np.array([r for r in range(len(x20)) if r not in randIdx])
    xTrain20 = x20[pRandIdx]
    yTrain20 = y20[pRandIdx]
    xValid20 = x20[randIdx]
    yValid20 = y20[randIdx]

    eValList = []
    for gamPower in gammaList:
        clf  = svm.SVC(C=0.1, kernel='rbf', gamma=(10 ** gamPower))
        eVal = 1 - clf.fit(xTrain20, yTrain20).score(xValid20, yValid20)
        eValList.append(eVal)
    return np.argmin(eValList)

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
    REPEAT = 100
    num_cores = mp.cpu_count()

    t0 = time.time()
    eValHist = Parallel(n_jobs=num_cores)(delayed(eValCal)() for i in range(REPEAT))
    plotHist(eValHist, r"$\log_{10}\gamma$", r'$E_{\mathrm{val}}$', "Q20", 1, False)
    eValHist = np.bincount(eValHist)
    eValHist = np.array(eValHist)
    # while len(eValHist) < 5:
    #     eValHist = np.append(eValHist, 0)
    t1 = time.time()

    print '========================================================='
    print 'Question 20:', eValHist
    print '---------------------------------------------------------'
    print 'Q20 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()