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

def nborCalc(k, xTraData, xTesData, yTraData, yTesData):
    hList = []
    for x in xTesData:
        hList.append(findKNbor(k, x, xTraData, yTraData))
    return np.mean((yTesData != hList).astype(int))

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

    TRAIN14_FILE = 'hw8_train.dat'
    TRAIN14_DATA = np.loadtxt(TRAIN14_FILE, dtype=np.float)
    xTra14 = TRAIN14_DATA[:, 0:(TRAIN14_DATA.shape[1] - 1)]
    yTra14 = TRAIN14_DATA[:, TRAIN14_DATA.shape[1] - 1]

    TEST14_FILE = 'hw8_test.dat'
    TEST14_DATA = np.loadtxt(TEST14_FILE, dtype=np.float)
    xTes14 = TEST14_DATA[:, 0:(TEST14_DATA.shape[1] - 1)]
    yTes14 = TEST14_DATA[:, (TEST14_DATA.shape[1] - 1)]

    eOutList = []
    for k in kList:
        eOutList.append(nborCalc(k, xTra14, xTes14, yTra14, yTes14))

    plotFig(kList, eOutList, r"$k$", r'$E_{\mathrm{out}}(g)$', "Q14", 1, False)

    t1 = time.time()

    print "============================================"
    print "Q14: Eout(g) =", np.mean(eOutList)
    print "--------------------------------------------"
    print "Q14 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()