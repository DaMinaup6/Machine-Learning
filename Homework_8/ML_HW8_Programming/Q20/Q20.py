import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from joblib import Parallel, delayed

def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0

def kMeans(k, xData):
    grArr = []
    muArr = xData[np.random.choice(xData.shape[0], k)]
    converge = False
    while not converge:
        grArr = []
        oriMu = np.copy(muArr)
        # optimize S
        for x in xData:
            dtArr = []
            for idx in range(k):
                dtArr.append([idx, np.linalg.norm(x - muArr[idx])])
            dtArr = np.array(dtArr)
            dtArr = dtArr[dtArr[:, 1].argsort()]
            grArr.append(dtArr[0][0])
        grArr = np.column_stack((np.array(grArr), xData))
        grArr = grArr[grArr[:, 0].argsort()]
        # optimize mu
        muNum = 1.0
        muIdx = grArr[0][0]
        muTmp = grArr[0][range(1, grArr.shape[1])]
        for row in range(1, grArr.shape[0]):
            if grArr[row][0] != muIdx:
                muArr[muIdx] = (muTmp / muNum)
                muNum = 1.0
                muIdx = grArr[row][0]
                muTmp = grArr[row][range(1, grArr.shape[1])]
            else:
                muNum += 1.0
                muTmp += grArr[row][range(1, grArr.shape[1])]
            if row == grArr.shape[0] - 1:
                muArr[muIdx] = (muTmp / muNum)
        if np.array_equal(muArr, oriMu):
            converge = True
    return errCalc(muArr, grArr)


def errCalc(muArr, grArr):
    errSum = 0.0
    for g in grArr:
        gp = g[0]
        x  = g[range(1, grArr.shape[1])]
        errSum += np.linalg.norm(x - muArr[gp]) ** 2
    return errSum / grArr.shape[0]

def plotFig(x, y, xLabel, yLabel, title, width, isFloat):
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
    REPEAT = 500
    kList  = [2, 4, 6, 8, 10]
    num_cores = mp.cpu_count()

    t0 = time.time()

    TRAIN20_FILE = 'hw8_nolabel_train.dat'
    TRAIN20_DATA = np.loadtxt(TRAIN20_FILE, dtype=np.float)
    xTra20       = TRAIN20_DATA

    eInList = []
    for k in kList:
        eIn = Parallel(n_jobs=num_cores)(delayed(kMeans)(k, xTra20) for i in range(REPEAT))
        eInList.append(np.mean(eIn))

    plotFig(kList, eInList, r"$k$", r'$E_{\mathrm{in}}$', "Q20", 1, False)

    t1 = time.time()

    print "============================================"
    print "Q20: Ein(g) =", np.mean(eInList)
    print "--------------------------------------------"
    print "Q20 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()