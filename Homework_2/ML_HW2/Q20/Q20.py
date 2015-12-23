import numpy as np
import random as rd
import math
import time

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

def mulDSA(x, y, H):
    einErr = []
    hList = []
    dataList = []
    for xI in range(len(x[0])):
        xArr = x[:, xI]
        data = np.column_stack((xArr, y))
        data = np.array(sorted(data, key=lambda data: data[0]), dtype=np.float)
        (hArr, err) = errIn(data[:, 1], H)
        einErr.append(err)
        hList.append(hArr)
        dataList.append(data)
    eMin = np.argmin(einErr)
    return (einErr[eMin], hList[eMin], dataList[eMin])

def errIn(y, H):
    result = []
    for i in range(len(H)):
        yErr = H[i][H[i] != y]
        result.append(float(yErr.shape[0]))
    minInd = np.argmin(result)
    return (H[minInd], result[minInd] / len(y))

def sTheta(data, indexMin, indexMax):
    if indexMax > indexMin:
        (s, theta) = (1.0,  (data[indexMax][0] + data[indexMax - 1][0]) / 2)
    elif indexMax < indexMin:
        (s, theta) = (-1.0, (data[indexMin][0] + data[indexMin - 1][0]) / 2)
    else:
        if data[indexMin][1] == 1.0:
            (s, theta) = (1.0, data[indexMin][0])
        elif data[indexMin][0] == -1.0:
            (s, theta) = (-1.0, data[indexMin][0])
    return (s, theta)

def errOut(s, theta, x, y):
    errOut = []
    for xI in range(len(x[0])):
        xArr = x[:, xI]
        data = np.column_stack((xArr, y))
        data = np.array(sorted(data, key=lambda data: data[0]), dtype=np.float)
        h = []
        for i in range(len(xArr)):
            h.append(float(s * sign(xArr[i] - theta)))
        h = np.array(h, dtype=np.float)
        yErr = h[h != y]
        errOut.append(float(yErr.shape[0]) / len(y))
    return min(errOut)

def main():
    t0 = time.time()

    TRAIN_FILE = "hw2_train.dat"
    TRAIN_DATA = np.loadtxt(TRAIN_FILE, dtype=np.float)

    TEST_FILE = "hw2_test.dat"
    TEST_DATA = np.loadtxt(TEST_FILE, dtype=np.float)

    x = TRAIN_DATA[:, 0:9]
    y = TRAIN_DATA[:, 9]
    H = HGenetation(len(y))
    (eIn, h, data) = mulDSA(x, y, H)
    (s, theta) = sTheta(data, np.argmin(h), np.argmax(h))

    x = TEST_DATA[:, 0:9]
    y = TEST_DATA[:, 9]
    eOut = errOut(s, theta, x, y)

    t1 = time.time()

    print "====================================="
    print "Q20: ", eOut
    print "-------------------------------------"
    print "Q20 costs ", t1 - t0, ' seconds'
    print "====================================="

if __name__ == '__main__':
    main()