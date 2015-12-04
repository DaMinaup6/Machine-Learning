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

    TRAIN15_FILE = 'hw5_train.dat'
    TRAIN15_DATA = np.loadtxt(TRAIN15_FILE, dtype=np.float)
    x15 = TRAIN15_DATA[:, 1:TRAIN15_DATA.shape[1]]
    y15 = digitTrans(TRAIN15_DATA[:, 0], DIGIT)

    CList = [-6, -4, -2, 0, 2]
    wList = []
    t0 = time.time()
    for cPower in range(-6, 3, 2):
        clf = svm.SVC(C=(10 ** cPower), kernel='linear')
        clf.fit(x15, y15)
        w     = clf.coef_[0]
        wNorm = np.linalg.norm(w)
        wList.append(wNorm)
    plotHist(CList, wList, r"$\log_{10}C$", r'$\Vert\bf{w}\Vert$', "Q15", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 15:', wList
    print '---------------------------------------------------------'
    print 'Q15 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()