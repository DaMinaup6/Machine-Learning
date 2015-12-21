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

def tGEinCal(x, y, gList, aList, eInList):
    T = len(gList)
    GEinList = []
    for t in range(T + 1):
        if t != 0:
            gEinList = []
            for j in range(len(x)):
                gS = 0.0
                for k in range(t):
                    (i, s, theta) = gList[k]
                    gS += aList[k] * deciStump(x[j][i], s, theta)
                gEinList.append(sign(gS))
            gEinList = np.array(gEinList)
            GEinList.append(1 - float(np.sum(gEinList == y)) / len(y))
    return GEinList


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

    TRAIN14_FILE = 'hw2_adaboost_train.dat'
    TRAIN14_DATA = np.loadtxt(TRAIN14_FILE, dtype=np.float)
    x14 = TRAIN14_DATA[:, 0:(TRAIN14_DATA.shape[1] - 1)]
    y14 = TRAIN14_DATA[:, (TRAIN14_DATA.shape[1] - 1)]
    H = HGenetation(len(y14))

    (gList, aList, eInList) = adaBoost(x14, y14, H, T)
    tGEinList = tGEinCal(x14, y14, gList, aList, eInList)
    tList = np.array(range(T)) + 1
    plotHist(tList, tGEinList, r"$t$", r'$E_{\mathrm{in}}(G_t)$', "Q14", 1, False)

    t1 = time.time()

    print "============================================"
    print "Q14:", "Ein(G) =", tGEinList[-1]
    print "--------------------------------------------"
    print "Q14 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()