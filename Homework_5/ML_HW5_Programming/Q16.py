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
    DIGIT = 8

    TRAIN16_FILE = 'hw5_train.dat'
    TRAIN16_DATA = np.loadtxt(TRAIN16_FILE, dtype=np.float)
    x16 = TRAIN16_DATA[:, 1:TRAIN16_DATA.shape[1]]
    y16 = digitTrans(TRAIN16_DATA[:, 0], DIGIT)

    CList   = [-6, -4, -2, 0, 2]
    eInList = []
    t0 = time.time()
    for cPower in CList:
        clf = svm.SVC(C=(10 ** cPower), kernel='poly', degree=2, gamma=1, coef0=1)
        eIn = 1 - clf.fit(x16, y16).score(x16, y16)
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