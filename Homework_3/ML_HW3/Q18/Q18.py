import math
import time
import numpy as np

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

def main():
	eta = 0.001
	Times = 2000

	t0 = time.time()
	TRAIN18_FILE = 'hw3_train.dat'
	TRAIN18_DATA = np.loadtxt(TRAIN18_FILE, dtype=np.float)
	xTrain18 = np.column_stack((np.ones(TRAIN18_DATA.shape[0]), TRAIN18_DATA[:, 0:(TRAIN18_DATA.shape[1] - 1)]))
	yTrain18 = TRAIN18_DATA[:, (TRAIN18_DATA.shape[1] - 1)]

	TEST18_FILE = 'hw3_test.dat'
	TEST18_DATA = np.loadtxt(TEST18_FILE, dtype=np.float)
	xTest18 = np.column_stack((np.ones(TEST18_DATA.shape[0]), TEST18_DATA[:, 0:(TEST18_DATA.shape[1] - 1)]))
	yTest18 = TEST18_DATA[:, (TEST18_DATA.shape[1] - 1)]

	w18 = GDA(xTrain18, yTrain18, eta, Times)
	eOut = errRate(xTest18, yTest18, w18)
	t1 = time.time()
	print '========================================================='
	print 'Question 18:', eOut, 'with w', w18
	print '---------------------------------------------------------'
	print 'Q18 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()