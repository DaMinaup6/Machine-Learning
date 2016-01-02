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

def printTree(tree, tab, root, left):
    spaceStr = ""
    for t in range(tab):
        spaceStr += "\t"
    if root:
        spaceStr += "root "
    else:    
        spaceStr += "left  " if left else "right "
    if tree.data is not None:
        print spaceStr + str(tree.data)
    if tree.left is not None:
        printTree(tree.left,  tab + 1, False, True)
    if tree.right is not None:
        printTree(tree.right, tab + 1, False, False)

def main():
    t0 = time.time()

    TRAIN13_FILE = 'hw3_train.dat'
    TRAIN13_DATA = np.loadtxt(TRAIN13_FILE, dtype=np.float)
    xTra13 = TRAIN13_DATA[:, 0:(TRAIN13_DATA.shape[1] - 1)]
    yTra13 = TRAIN13_DATA[:, TRAIN13_DATA.shape[1] - 1]
    
    Nprime   = len(TRAIN13_DATA)
    deciTree = decisionTree(TRAIN13_DATA, 0, 0, 1.0, 0.0, Tree())
    printTree(deciTree, 0, True, False)

    t1 = time.time()

    print "============================================"
    print "Q13:"
    print "--------------------------------------------"
    print "Q13 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()