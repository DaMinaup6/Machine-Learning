import time
import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0

def findKNbor(k, xIpt, xData, yData):
    nborNum  = 0.0
    distData = []
    for xIdx in range(xData.shape[0]):
        distData.append([xIdx, np.linalg.norm(xIpt - xData[xIdx])])
    distData = np.array(distData)
    distData = distData[distData[:, 1].argsort()]
    for num in range(k):
        nborNum += yData[distData[num][0]]
    return sign(nborNum)

def nborCalc(k, xData, yData):
    hList = []
    for x in xData:
        hList.append(findKNbor(k, x, xData, yData))
    return np.mean((yData != hList).astype(int))

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
    kList = [1, 3, 5, 7, 9]

    t0 = time.time()

    TRAIN12_FILE = 'hw8_train.dat'
    TRAIN12_DATA = np.loadtxt(TRAIN12_FILE, dtype=np.float)
    xTra12 = TRAIN12_DATA[:, 0:(TRAIN12_DATA.shape[1] - 1)]
    yTra12 = TRAIN12_DATA[:, TRAIN12_DATA.shape[1] - 1]

    TEST12_FILE = 'hw8_test.dat'
    TEST12_DATA = np.loadtxt(TEST12_FILE, dtype=np.float)
    xTes12 = TEST12_DATA[:, 0:(TEST12_DATA.shape[1] - 1)]
    yTes12 = TEST12_DATA[:, (TEST12_DATA.shape[1] - 1)]

    eInList = []
    for k in kList:
        eInList.append(nborCalc(k, xTra12, yTra12))

    plotFig(kList, eInList, r"$k$", r'$E_{\mathrm{in}}(g)$', "Q12", 1, False)

    t1 = time.time()

    print "============================================"
    print "Q12: Ein(g) =", np.mean(eInList)
    print "--------------------------------------------"
    print "Q12 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()