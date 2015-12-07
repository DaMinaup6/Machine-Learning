import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def digitTrans(y, digit):
    y[y != digit] = -1.0
    y[y == digit] = 1.0
    return y

def plotHist(x, y, xLabel, yLabel, title, width, isFloat):
    plt.title(str(title))
    plt.xlabel(str(xLabel))
    plt.ylabel(str(yLabel))
    if isFloat: plt.hist(x)
    else: plt.plot(x, y)
    plt.grid(True)
    plt.draw()
    plt.savefig(title)
    plt.close()

def main():
    DIGIT = 0

    TRAIN19_FILE = 'hw5_train.dat'
    TRAIN19_DATA = np.loadtxt(TRAIN19_FILE, dtype=np.float)
    xTrain19 = TRAIN19_DATA[:, 1:TRAIN19_DATA.shape[1]]
    yTrain19 = digitTrans(TRAIN19_DATA[:, 0], DIGIT)

    TEST19_FILE = 'hw5_test.dat'
    TEST19_DATA = np.loadtxt(TEST19_FILE, dtype=np.float)
    xTest19 = TEST19_DATA[:, 1:TEST19_DATA.shape[1]]
    yTest19 = digitTrans(TEST19_DATA[:, 0], DIGIT)

    gammaList = [0, 1, 2, 3, 4]
    eOutList  = []
    t0 = time.time()
    for gamPower in gammaList:
        clf  = svm.SVC(C=0.1, kernel='rbf', gamma=(10 ** gamPower))
        eOut = 1 - clf.fit(xTrain19, yTrain19).score(xTest19, yTest19)
        eOutList.append(eOut)
    plotHist(gammaList, eOutList, r"$\log_{10}\gamma$", r'$E_{\mathrm{out}}$', "Q19", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 19:', eOutList
    print '---------------------------------------------------------'
    print 'Q19 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()