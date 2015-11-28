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
    DIGIT = 8

    TRAIN17_FILE   = 'hw5_train.dat'
    TRAIN17_DATA_X = np.loadtxt(TRAIN17_FILE, dtype=np.float)
    TRAIN17_DATA_Y = np.loadtxt(TRAIN17_FILE, dtype=np.float)

    x17 = TRAIN17_DATA_X[:, 1:TRAIN17_DATA_X.shape[1]]
    y17 = TRAIN17_DATA_Y[:, 0:1]
    y17[y17 != DIGIT] = -1.0
    y17[y17 == DIGIT] = 1.0
    yArr = []
    for y in y17:
        yArr.append(y[0])
    y17 = np.array(yArr)

    CList   = []
    sumList = []
    t0 = time.time()
    for cPower in range(-6, 3, 2):
        clf = svm.SVC(C=0.01, kernel='poly', degree=2, gamma=1, coef0=1)
        clf.fit(x17, y17)
        sumNum = np.sum(np.absolute(clf.dual_coef_[0]))
        CList.append(cPower)
        sumList.append(sumNum)
    plotHist(CList, sumList, r"$\log_{10}C$", r'$\sum_{n=1}^N\alpha_n$', "Q17", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 17:', sumList
    print '---------------------------------------------------------'
    print 'Q17 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()