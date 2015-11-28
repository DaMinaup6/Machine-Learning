import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def plotHist(x, y, xLabel, yLabel, title, width, isFloat):
    plt.title(str(title))
    plt.xlabel(str(xLabel))
    plt.ylabel(str(yLabel))
    if isFloat: plt.hist(x)
    else:
        plt.plot(x, y)
    plt.grid(True)
    plt.draw()
    plt.savefig(title)
    plt.close()

def main():
    DIGIT = 0

    TRAIN19_FILE   = 'hw5_train.dat'
    TRAIN19_DATA_X = np.loadtxt(TRAIN19_FILE, dtype=np.float)
    TRAIN19_DATA_Y = np.loadtxt(TRAIN19_FILE, dtype=np.float)

    xTrain19 = TRAIN19_DATA_X[:, 1:TRAIN19_DATA_X.shape[1]]
    yTrain19 = TRAIN19_DATA_Y[:, 0:1]
    yTrain19[yTrain19 != DIGIT] = -1.0
    yTrain19[yTrain19 == DIGIT] = 1.0
    yArr = []
    for y in yTrain19:
        yArr.append(y[0])
    yTrain19 = np.array(yArr)

    TEST19_FILE   = 'hw5_test.dat'
    TEST19_DATA_X = np.loadtxt(TEST19_FILE, dtype=np.float)
    TEST19_DATA_Y = np.loadtxt(TEST19_FILE, dtype=np.float)

    xTest19 = TEST19_DATA_X[:, 1:TEST19_DATA_X.shape[1]]
    yTest19 = TEST19_DATA_Y[:, 0:1]
    yTest19[yTest19 != DIGIT] = -1.0
    yTest19[yTest19 == DIGIT] = 1.0
    yArr = []
    for y in yTest19:
        yArr.append(y[0])
    yTest19 = np.array(yArr)

    gamList  = []
    eOutList = []
    t0 = time.time()
    for gamPower in range(0, 5):
        clf  = svm.SVC(C=0.1, kernel='rbf', gamma=(10 ** gamPower))
        clf.fit(xTrain19, yTrain19)
        eOut = 1 - clf.fit(xTrain19, yTrain19).score(xTest19, yTest19)
        gamList.append(gamPower)
        eOutList.append(eOut)
    plotHist(gamList, eOutList, r"$\log_{10}\gamma$", r'$E_{\mathrm{out}}$', "Q19", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 19:', eOutList
    print '---------------------------------------------------------'
    print 'Q19 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()