import time
import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0

def gUniform(gamma, xIpt, xData, yData):
    gNum  = 0.0
    for idx in range(xData.shape[0]):
        gNum += yData[idx] * np.exp(-gamma * (np.linalg.norm(xIpt - xData[idx]) ** 2))
    return sign(gNum)

def gUniCalc(gamma, xTraData, xTesData, yTraData, yTesData):
    hList = []
    for x in xTesData:
        hList.append(gUniform(gamma, x, xTraData, yTraData))
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
    gList = [0.001, 0.1, 1, 10, 100]
    gLogList = [-3, -1, 0, 1, 2]

    t0 = time.time()

    TRAIN18_FILE = 'hw8_train.dat'
    TRAIN18_DATA = np.loadtxt(TRAIN18_FILE, dtype=np.float)
    xTra18 = TRAIN18_DATA[:, 0:(TRAIN18_DATA.shape[1] - 1)]
    yTra18 = TRAIN18_DATA[:, TRAIN18_DATA.shape[1] - 1]

    TEST18_FILE = 'hw8_test.dat'
    TEST18_DATA = np.loadtxt(TEST18_FILE, dtype=np.float)
    xTes18 = TEST18_DATA[:, 0:(TEST18_DATA.shape[1] - 1)]
    yTes18 = TEST18_DATA[:, (TEST18_DATA.shape[1] - 1)]

    eOutList = []
    for gamma in gList:
        eOutList.append(gUniCalc(gamma, xTra18, xTes18, yTra18, yTes18))

    plotFig(gList, eOutList, r"$\gamma$", r'$E_{\mathrm{out}}(g)$', "Q18", 1, False)

    t1 = time.time()

    print "============================================"
    print "Q18: Eout(g) =", np.mean(eOutList)
    print "--------------------------------------------"
    print "Q18 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()