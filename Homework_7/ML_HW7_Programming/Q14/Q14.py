import time
import numpy as np
from collections import Counter

class Tree(object):
    def __init__(self):
        self.left  = None
        self.right = None
        self.data  = None

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

def treeCalErr(X, Y, deciTree):
    y = []
    for x in X:
        y.append(treePredict(x, deciTree))
    return np.mean((y != Y).astype(int))

def main():
    t0 = time.time()

    TRAIN14_FILE = 'hw3_train.dat'
    TRAIN14_DATA = np.loadtxt(TRAIN14_FILE, dtype=np.float)
    xTra14 = TRAIN14_DATA[:, 0:(TRAIN14_DATA.shape[1] - 1)]
    yTra14 = TRAIN14_DATA[:, TRAIN14_DATA.shape[1] - 1]
    
    deciTree = decisionTree(TRAIN14_DATA, 0, 0, 1.0, 0.0, Tree())

    t1 = time.time()

    print "============================================"
    print "Q14: Ein =", treeCalErr(xTra14, yTra14, deciTree)
    print "--------------------------------------------"
    print "Q14 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()