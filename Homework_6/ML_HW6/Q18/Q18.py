import time
import numpy as np
import matplotlib.pyplot as plt

def deciStump(x, s, theta):
    return s * sign(x - theta)

def sign(x):
	if x >= 0:
		return +1.0
	else:
		return -1.0

def HGenetation(N):
    H = np.ones((N, N), dtype=np.float)
    index = 1
    for i in range(1, N):
        for j in range(0, index):
            H[j][i] = -1.0
        index += 1
    return np.concatenate((H, -H))

def mulDSA(x, y, H, u):
    einUErr  = []
    hList    = []
    dataList = []
    for xI in range(len(x[0])):
        xArr = x[:, xI]
        data = np.column_stack((u, xArr, y))
        data = np.array(sorted(data, key=lambda data: data[1]), dtype=np.float)
        (hArr, eU) = errInU(data[:, (data.shape[1] - 1)], H, data[:, 0])
        einUErr.append(eU)
        hList.append(hArr)
        dataList.append(data)
    eMinIdx = np.argmin(einUErr)
    return (eMinIdx, einUErr[eMinIdx], hList[eMinIdx], dataList[eMinIdx])

def errInU(y, H, u):
    eUList = []
    for i in range(len(H)):
        eUList.append(errU(y, H[i], u))
    minInd = np.argmin(eUList)
    return (H[minInd], min(eUList))

def errU(y, yS, u):
    yErr = (yS != y).astype(float)
    return float(np.sum(np.dot(np.transpose(u), yErr))) / np.sum(u)

def sTheta(data, indexMin, indexMax):
    if indexMax > indexMin:
        (s, theta) = (1.0,  (data[indexMax][1] + data[indexMax - 1][1]) / 2)
    elif indexMax < indexMin:
        (s, theta) = (-1.0, (data[indexMin][1] + data[indexMin - 1][1]) / 2)
    else:
        if data[indexMin][1] == 1.0:
            (s, theta) = (1.0, data[indexMin][1])
        elif data[indexMin][0] == -1.0:
            (s, theta) = (-1.0, data[indexMin][1])
    return (s, theta)

def errIn(s, theta, xArr, y):
    yS = []
    for i in range(len(xArr)):
        yS.append(float(s * sign(xArr[i] - theta)))
    yS = np.array(yS, dtype=np.float)
    return (yS != y).astype(int)

def adaBoost(x, y, H, T):
    N     = len(y)
    uList = np.ones(N) / N

    gList, aList, eInList = [], [], []
    for t in range(T):
        (i, eInU, h, data) = mulDSA(x, y, H, uList)
        (s, theta) = sTheta(data, np.argmin(h), np.argmax(h))
        upCoe = np.sqrt((1 - eInU) / eInU)
        alpha = np.log(upCoe)

        xArr = x[:, i]
        data = np.column_stack((xArr, y))
        data = np.array(sorted(data, key=lambda data: data[0]), dtype=np.float)
        uErr = errIn(s, theta, xArr, y)
        for u in range(len(uErr)):
            if uErr[u]:
                uList[u] *= upCoe
            else:
                uList[u] /= upCoe
        gList.append((i, s, theta))
        aList.append(alpha)
        eInList.append(float(np.sum(data[:, 1] != h)) / len(h))

    return (gList, aList, eInList)

def errOut(s, theta, x, y, i):
    errOut = []
    xArr = x[:, i]
    data = np.column_stack((xArr, y))
    data = np.array(sorted(data, key=lambda data: data[0]), dtype=np.float)
    h = []
    for i in range(len(xArr)):
        h.append(float(s * sign(xArr[i] - theta)))
    h = np.array(h, dtype=np.float)
    yErr = h[h != y]
    errOut.append(float(yErr.shape[0]) / len(y))
    return min(errOut)

def eOutCal(gList, x, y):
    eOutErr = []
    for g in gList:
        eOutErr.append(errOut(g[1], g[2], x, y, g[0]))
    return eOutErr

def tGEoutCal(x, y, gList, aList, eOutList):
    T = len(gList)
    GEoutList = []
    for t in range(T + 1):
        if t != 0:
            gEoutList = []
            for j in range(len(x)):
                gS = 0.0
                for k in range(t):
                    (i, s, theta) = gList[k]
                    gS += aList[k] * deciStump(x[j][i], s, theta)
                gEoutList.append(sign(gS))
            gEoutList = np.array(gEoutList)
            GEoutList.append(1 - float(np.sum(gEoutList == y)) / len(y))
    return GEoutList

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
    T = 300

    t0 = time.time()

    TRAIN18_FILE = 'hw2_adaboost_train.dat'
    TRAIN18_DATA = np.loadtxt(TRAIN18_FILE, dtype=np.float)
    xTra18 = TRAIN18_DATA[:, 0:(TRAIN18_DATA.shape[1] - 1)]
    yTra18 = TRAIN18_DATA[:, (TRAIN18_DATA.shape[1] - 1)]
    H = HGenetation(len(yTra18))

    TEST18_FILE = 'hw2_adaboost_test.dat'
    TEST18_DATA = np.loadtxt(TEST18_FILE, dtype=np.float)
    xTes18 = TEST18_DATA[:, 0:(TEST18_DATA.shape[1] - 1)]
    yTes18 = TEST18_DATA[:, (TEST18_DATA.shape[1] - 1)]

    (gList, aList, eInList) = adaBoost(xTra18, yTra18, H, T)
    eOutList = eOutCal(gList, xTes18, yTes18)
    tGEoutList = tGEoutCal(xTes18, yTes18, gList, aList, eOutList)
    tList = np.array(range(T)) + 1
    plotHist(tList, tGEoutList, r"$t$", r'$E_{\mathrm{out}}(G_t)$', "Q18", 1, False)

    t1 = time.time()

    print "============================================"
    print "Q18:", "Eout(G) =", tGEoutList[-1]
    print "--------------------------------------------"
    print "Q18 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()