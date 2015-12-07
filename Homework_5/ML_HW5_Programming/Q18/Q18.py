import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def digitTrans(y, digit):
    y[y != digit] = -1.0
    y[y == digit] = 1.0
    return y

def gauKernel(x1, x2, gamma):
    return np.exp(-1 * gamma * (np.linalg.norm(x1 - x2) ** 2))

def svmMargin(supVec, dualCoef, gamma):
    kerMat = np.matrix([[gauKernel(x1, x2, gamma) for x2 in supVec] for x1 in supVec])
    return reduce(np.dot, [dualCoef, kerMat, dualCoef])

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
    GAMMA = 100

    TRAIN18_FILE = 'hw5_train.dat'
    TRAIN18_DATA = np.loadtxt(TRAIN18_FILE, dtype=np.float)
    x18 = TRAIN18_DATA[:, 1:TRAIN18_DATA.shape[1]]
    y18 = digitTrans(TRAIN18_DATA[:, 0], DIGIT)

    CList    = [-3, -2, -1, 0, 1]
    distList = []
    t0 = time.time()
    for cPower in CList:
        clf = svm.SVC(C=(10 ** cPower), kernel='rbf', gamma=GAMMA)
        clf.fit(x18, y18)
        wSquare = svmMargin(np.array(clf.support_vectors_), np.array(clf.dual_coef_[0]), GAMMA)
        distList.append(1 / np.sqrt(wSquare.item(0)))
    plotHist(CList, distList, r"$\log_{10}C$", 'distance', "Q18", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 18:', distList
    print '---------------------------------------------------------'
    print 'Q18 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()