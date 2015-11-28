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
    TRAIN15_FILE = 'hw5_train.dat'
    TRAIN15_DATA_X = np.loadtxt(TRAIN15_FILE, dtype=np.float)
    TRAIN15_DATA_Y = np.loadtxt(TRAIN15_FILE, dtype=np.float)

    x15 = TRAIN15_DATA_X[:, 1:TRAIN15_DATA_X.shape[1]]
    y15 = TRAIN15_DATA_Y[:, 0:1]
    y15[y15 != 0] = -1.0
    y15[y15 == 0] = 1.0
    yArr = []
    for y in y15:
        yArr.append(y[0])
    y15 = np.array(yArr)

    CList = []
    wList = []
    t0 = time.time()
    for cPower in range(-6, 3, 2):
        clf = svm.SVC(C=(10 ** cPower), kernel='linear')
        clf.fit(x15, y15)
        w = clf.coef_[0]
        wNorm = np.linalg.norm(w)
        CList.append(cPower)
        wList.append(wNorm)
    plotHist(CList, wList, r"$\log_{10}C$", r'$||\bf{w}||$', "Q15", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 15:', wList
    print '---------------------------------------------------------'
    print 'Q15 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()