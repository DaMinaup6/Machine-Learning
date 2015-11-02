import math
import time
import numpy as np
import matplotlib.pyplot as plt

def theta(num):
	return 1 / (1 + np.exp(-num))

def grad(xArr, yArr, w):
	N = len(yArr)
	sumUp = np.zeros(xArr.shape[1])
	for i in range(N):
		sumUp += theta(-yArr[i] * np.dot(w, xArr[i])) * (-yArr[i] * xArr[i])
	return sumUp / N

def GDA(x, y, eta, Times):
	w = np.zeros(x.shape[1])
	times = 0
	while times < Times:
		g = grad(x, y, w)
		w -= eta * g
		times += 1
	return w

def errRate(x, y, w):
    yS = np.dot(x, w)
    yS[yS <= 0] = -1.0
    yS[yS > 0] = 1.0
    yErr = yS[yS != y]
    errCount = yErr.shape[0]
    return float(errCount) / len(y)

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

def main():
	eta = 0.01
	Times = 2000

	eOutList = []
	t0 = time.time()
	TRAIN19_FILE = 'hw3_train.dat'
	TRAIN19_DATA = np.loadtxt(TRAIN19_FILE, dtype=np.float)
	xTrain19 = np.column_stack((np.ones(TRAIN19_DATA.shape[0]), TRAIN19_DATA[:, 0:(TRAIN19_DATA.shape[1] - 1)]))
	yTrain19 = TRAIN19_DATA[:, (TRAIN19_DATA.shape[1] - 1)]

	TEST19_FILE = 'hw3_test.dat'
	TEST19_DATA = np.loadtxt(TEST19_FILE, dtype=np.float)
	xTest19 = np.column_stack((np.ones(TEST19_DATA.shape[0]), TEST19_DATA[:, 0:(TEST19_DATA.shape[1] - 1)]))
	yTest19 = TEST19_DATA[:, (TEST19_DATA.shape[1] - 1)]

	for times in range(repeat):
		w19 = GDA(xTrain19, yTrain19, eta, Times)
		eOutList.append(errRate(xTest19, yTest19, w19))
	eOutAve = sum(eOutList) / float(repeat)
	plotHist(eOutList, "Eout Error Rate", "Frequency", "Q19", 0.01, True)
	t1 = time.time()
	print '========================================================='
	print 'Question 19:', eOutAve
	print '---------------------------------------------------------'
	print 'Q19 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()