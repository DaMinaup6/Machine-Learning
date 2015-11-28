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

    TRAIN18_FILE   = 'hw5_train.dat'
    TRAIN18_DATA_X = np.loadtxt(TRAIN18_FILE, dtype=np.float)
    TRAIN18_DATA_Y = np.loadtxt(TRAIN18_FILE, dtype=np.float)

    x18 = TRAIN18_DATA_X[:, 1:TRAIN18_DATA_X.shape[1]]
    y18 = TRAIN18_DATA_Y[:, 0:1]
    y18[y18 != DIGIT] = -1.0
    y18[y18 == DIGIT] = 1.0
    yArr = []
    for y in y18:
        yArr.append(y[0])
    y18 = np.array(yArr)

    CList   = []
    disList = []
    t0 = time.time()
    for cPower in range(-3, 2):
        clf = svm.SVC(C=(10 ** cPower), kernel='rbf', gamma=100)
        clf.fit(x18, y18)

        w = clf.dual_coef_.dot(clf.support_vectors_)[0]
        margin = 2 / np.sqrt((w ** 2).sum())
        # for idx, val in enumerate(clf.dual_coef_[0]):
        #     dist = 0
        #     if np.absolute(val) < (10 ** cPower) and np.absolute(val) > 0:
        #         dist = np.linalg.norm(x18[clf.support_[idx]] - w)
        CList.append(cPower)
        disList.append(margin)
    plotHist(CList, disList, r"$\log_{10}C$", 'margin', "Q18", 1, False)
    t1 = time.time()
    print '========================================================='
    print 'Question 18:', disList
    print '---------------------------------------------------------'
    print 'Q18 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()