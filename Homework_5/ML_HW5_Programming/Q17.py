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
    plt.tight_layout()
    plt.close()

def main():
    DIGIT = 8

    TRAIN17_FILE = 'hw5_train.dat'
    TRAIN17_DATA = np.loadtxt(TRAIN17_FILE, dtype=np.float)
    x17 = TRAIN17_DATA[:, 1:TRAIN17_DATA.shape[1]]
    y17 = digitTrans(TRAIN17_DATA[:, 0], DIGIT)

    CList   = [-6, -4, -2, 0, 2]
    sumList = []
    t0 = time.time()
    for cPower in CList:
        clf = svm.SVC(C=(10 ** cPower), kernel='poly', degree=2, gamma=1, coef0=1)
        clf.fit(x17, y17)
        sumNum = np.sum(np.absolute(clf.dual_coef_[0]))
        sumList.append(sumNum)
    plotHist(CList, sumList, r"$\log_{10}C$", r'$\sum\alpha_n$', "Q17", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 17:', sumList
    print '---------------------------------------------------------'
    print 'Q17 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()