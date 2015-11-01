import numpy as np
import random as rd
import math
import time
import matplotlib.pyplot as plt

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

def dataGeneration(dataSize):
	data = []
	for times in range(dataSize):
		item = []
		ranNum = rd.uniform(-1.0, 1.0)
		ySign = sign(ranNum)
		flip = rd.uniform(0.0, 1.0) < 0.2
		item.append(ranNum)
		if flip:
			item.append(-ySign)
		else:
			item.append(ySign)
		data.append(item)
	return np.array(sorted(data, key=lambda data: data[0]), dtype=np.float)

def repeatDSA(H, DATA_SIZE, noise, times):
    totEinErr = 0.0
    totEoutErr = 0.0
    eInList = []
    eOutList = []
    for time in range(times):
        data = dataGeneration(DATA_SIZE)
        (hArr, err) = errIn(data[:, 1], H)
        (s, theta) = sTheta(data, np.argmin(hArr), np.argmax(hArr))
        errOut = theoErrOut(s, theta, noise)
        totEinErr += err
        totEoutErr += errOut
        eInList.append(err)
        eOutList.append(errOut)
    return (totEinErr / times, totEoutErr / times, eInList, eOutList)

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
    return (einErr[eMin], hList[eMin], dataList[eMin], eMin)

def sTheta(data, indexMin, indexMax):
    if indexMax > indexMin:
        (s, theta) = (1.0, data[indexMax][0])
    elif indexMax < indexMin:
        (s, theta) = (-1.0, data[indexMin][0])
    else:
        if data[indexMin][1] == 1:
            (s, theta) = (1.0, -1.0)
        else:
            (s, theta) = (-1.0, -1.0)
    return (s, theta)

def errIn(y, H):
    result = []
    for i in range(len(H)):
        yErr = H[i][H[i] != y]
        result.append(float(yErr.shape[0]))
    minInd = np.argmin(result)
    return (H[minInd], result[minInd] / len(y))

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

def theoErrOut(s, theta, noise):
    return 0.5 + (0.5 - noise) * s * (math.fabs(theta) - 1)

def plotHist(x, xLabel, yLabel, title, width, isFloat):
    plt.title(str(title))
    plt.xlabel(str(xLabel))
    plt.ylabel(str(yLabel))
    if isFloat: plt.hist(x)
    else:
        freq = np.bincount(x)
        freqIndex = np.nonzero(freq)[0]
        plt.bar(freqIndex, freq[freqIndex], width)
    plt.grid(True)
    plt.draw()
    plt.savefig(title)
    plt.close()