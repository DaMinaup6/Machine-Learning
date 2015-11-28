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

def errRate(x, y, w):
    yS = np.dot(x, w)
    yS[yS <= 0] = -1.0
    yS[yS > 0] = 1.0
    yErr = yS[yS != y]
    errCount = yErr.shape[0]
    return float(errCount) / len(y)

def main():
    DIGIT = 8

    TRAIN16_FILE = 'hw5_train.dat'
    TRAIN16_DATA_X = np.loadtxt(TRAIN16_FILE, dtype=np.float)
    TRAIN16_DATA_Y = np.loadtxt(TRAIN16_FILE, dtype=np.float)

    x16 = TRAIN16_DATA_X[:, 1:TRAIN16_DATA_X.shape[1]]
    y16 = TRAIN16_DATA_Y[:, 0:1]
    y16[y16 != DIGIT] = -1.0
    y16[y16 == DIGIT] = 1.0
    yArr = []
    for y in y16:
        yArr.append(y[0])
    y16 = np.array(yArr)

    CList   = []
    eInList = []
    t0 = time.time()
    for cPower in range(-6, 3, 2):
        clf = svm.SVC(C=(10 ** cPower), kernel='poly', degree=2, gamma=1, coef0=1)
        clf.fit(x16, y16)
        eIn = 1 - clf.fit(x16, y16).score(x16, y16)
        CList.append(cPower)
        eInList.append(eIn)
    plotHist(CList, eInList, r"$\log_{10}C$", r'$E_{\mathrm{in}}$', "Q16", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 16:', eInList
    print '---------------------------------------------------------'
    print 'Q16 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()