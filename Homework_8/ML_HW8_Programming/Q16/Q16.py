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
    for x in xTraData:
        hList.append(gUniform(gamma, x, xTraData, yTraData))
    return np.mean((yTraData != hList).astype(int))

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

    TRAIN16_FILE = 'hw8_train.dat'
    TRAIN16_DATA = np.loadtxt(TRAIN16_FILE, dtype=np.float)
    xTra16 = TRAIN16_DATA[:, 0:(TRAIN16_DATA.shape[1] - 1)]
    yTra16 = TRAIN16_DATA[:, TRAIN16_DATA.shape[1] - 1]

    TEST16_FILE = 'hw8_test.dat'
    TEST16_DATA = np.loadtxt(TEST16_FILE, dtype=np.float)
    xTes16 = TEST16_DATA[:, 0:(TEST16_DATA.shape[1] - 1)]
    yTes16 = TEST16_DATA[:, (TEST16_DATA.shape[1] - 1)]

    eInList = []
    for gamma in gList:
        eInList.append(gUniCalc(gamma, xTra16, xTes16, yTra16, yTes16))

    plotFig(gLogList, eInList, r"$\log\gamma$", r'$E_{\mathrm{in}}(g)$', "Q16-log", 1, False)

    t1 = time.time()

    print "============================================"
    print "Q16: Ein(g) =", np.mean(eInList)
    print "--------------------------------------------"
    print "Q16 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()