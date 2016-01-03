import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class Tree(object):
    def __init__(self):
        self.left  = None
        self.right = None
        self.data  = None

def bootstrap(N, DATA):
    dataList = []
    for n in range(N):
        dataList.append(random.choice(DATA))
    return np.array(dataList)

def xySort(xyList):
    x = xyList[:, 0:(xyList.shape[1] - 1)]
    y = xyList[:, xyList.shape[1] - 1]

    dataList = []
    data     = np.column_stack((x, y))
    for xI in range(len(x[0])):
        data = data[data[:, xI].argsort()]
        dataList.append(data)
    return np.array(dataList)

def GiniIdx(yArr):
    N = len(yArr)
    totSum = 0.0
    for k in Counter(yArr):
        ySum = 0.0
        for y in yArr:
            if y == k:
                ySum += 1
        totSum += (ySum / N) ** 2
    return (1 - totSum)

def oneFeatBC(data):
    dsList = []
    y = data[:, data.shape[1] - 1]
    N = len(y)

    for n in range(N):
        dcNum1, dcNum2, impur1, impur2 = 0, 0, 0.0, 0.0
        if n == 0:
            dsList.append(N * GiniIdx(y))
        else:
            dcNum1 = n
            impur1 = GiniIdx(y[0:n])
            dcNum2 = N - n
            impur2 = GiniIdx(y[n:])
            dsList.append(dcNum1 * impur1 + dcNum2 * impur2)
    minIdx = np.argmin(dsList)
    return np.array([minIdx, dsList[minIdx]]) # return the index and min value of number times impurity of this dimension

def branchCrite(xyList):
    if xyList.shape[0] == 0:
        return (0, 0, 1.0, 0.0, np.zeros(1))

    dataList = xySort(xyList)

    deciList = []
    for idx in range(len(dataList)):
        deciList.append(oneFeatBC(dataList[idx]))
    deciList = np.array(deciList)
    fIdx  = np.argmin(deciList[:, 1]) # choosen feature index
    Idx   = deciList[fIdx][0]         # seperated index of feature dimension
    data  = dataList[fIdx]            # sorted (x, y)
    dataX = data[:, fIdx]
    dataY = data[:, data.shape[1] - 1]

    h = np.ones(len(dataY))
    theta = 0.0
    if Idx > 0: # 1-base
        theta = (dataX[Idx - 1] + dataX[Idx]) / 2
        for i in range(int(Idx)):
            h[i] = -1.0

    hPCor = np.mean((dataY != h).astype(int))
    hNCor = np.mean((dataY != -h).astype(int))
    s = 1.0 if hPCor > hNCor else -1.0
    h = h   if s == 1.0      else -h
    return (fIdx, Idx, s, theta, h)

def decisionTree(dataList, prefIdx, preIdx, preS, preTheta, root):
    (fIdx, Idx, s, theta, h) = branchCrite(dataList)
    if Idx == 0:
        return None
    else:
        root.data = (fIdx, s, theta)

        data   = dataList[dataList[:, fIdx].argsort()] # sort data by some feature
        d1, d2 = data[0:Idx, :], data[Idx:, :]         # seperate data into two parts

        root.left  = decisionTree(d1, fIdx, Idx, s, theta, Tree())
        root.right = decisionTree(d2, fIdx, Idx, s, theta, Tree())

        return root

def treePredict(x, deciTree):
    res = 0
    tree = deciTree
    while res == 0:
        fIdx  = tree.data[0]
        s     = tree.data[1]
        theta = tree.data[2]
        if x[fIdx] < theta:
            if tree.left:
                tree = tree.left
            else:
                res = s * 1.0
        else:
            if tree.right:
                tree = tree.right
            else:
                res = s * (-1.0)
    return res

def treeCalErr(X, Y, treeList):
    resList = []
    for tree in treeList:
        y = []
        for x in X:
            y.append(treePredict(x, tree))
        resList.append(np.mean((y != Y).astype(int)))
    return np.array(resList)

def bagAlg(T, N, DATA):
    treeList = []
    for t in range(T):
        D = bootstrap(N, DATA)
        (fIdx, Idx, s, theta, h) = branchCrite(D)
        deciTree = Tree()
        deciTree.data = (fIdx, s, theta)
        treeList.append(deciTree)
    return np.array(treeList)

def plotHist(x, y, xLabel, yLabel, title, width, isFloat):
    plt.figure(figsize=(18, 10))
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
    REPEAT = 100

    t0 = time.time()

    TRAIN20_FILE = 'hw3_train.dat'
    TRAIN20_DATA = np.loadtxt(TRAIN20_FILE, dtype=np.float)
    xTra20 = TRAIN20_DATA[:, 0:(TRAIN20_DATA.shape[1] - 1)]
    yTra20 = TRAIN20_DATA[:, TRAIN20_DATA.shape[1] - 1]

    TEST20_FILE = 'hw3_test.dat'
    TEST20_DATA = np.loadtxt(TEST20_FILE, dtype=np.float)
    xTes20 = TEST20_DATA[:, 0:(TEST20_DATA.shape[1] - 1)]
    yTes20 = TEST20_DATA[:, (TEST20_DATA.shape[1] - 1)]
    
    Nprime   = len(TRAIN20_DATA)
    treeList = []
    for times in range(REPEAT):
        treeList = np.append(treeList, bagAlg(T, Nprime, TRAIN20_DATA))

    pdList = []
    for tree in treeList:
        resList = []
        for x in xTes20:
            resList.append(treePredict(x, tree))
        pdList.append(np.array(resList))

    N = len(yTes20)
    EGtList = []
    idx = 0
    arr = np.zeros(N)
    while idx < (T * REPEAT):
        arr = np.add(arr, pdList[idx])
        pd  = np.copy(arr)
        pd[pd >= 0] = 1.0
        pd[pd <  0] = -1.0
        EGtList.append(np.mean((pd != yTes20).astype(int)))
        idx += 1

    repeatList = np.array(range(T * REPEAT)) + 1
    
    plotHist(repeatList, EGtList, r"$t$", r'$E_{\mathrm{in}}(G_t)$', "Q20", 1, False)

    t1 = time.time()

    print "============================================"
    print "Q20: Eout =", np.mean(EGtList)
    print "--------------------------------------------"
    print "Q20 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()