import math
import time
import numpy as np

def theta(num):
	return 1 / (1 + np.exp(-num))

def SGD(x, y, eta, Times):
	N = len(y)
	w = np.zeros(x.shape[1])
	times = 0
	while times < Times:
		for i in range(N):
			g = theta(-y[i] * np.dot(w, x[i])) * (y[i] * x[i])
			w += eta * g
			times += 1
			if times >= Times:
				break
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
	TRAIN20_FILE = 'hw3_train.dat'
	TRAIN20_DATA = np.loadtxt(TRAIN20_FILE, dtype=np.float)
	xTrain20 = np.column_stack((np.ones(TRAIN20_DATA.shape[0]), TRAIN20_DATA[:, 0:(TRAIN20_DATA.shape[1] - 1)]))
	yTrain20 = TRAIN20_DATA[:, (TRAIN20_DATA.shape[1] - 1)]

	TEST20_FILE = 'hw3_test.dat'
	TEST20_DATA = np.loadtxt(TEST20_FILE, dtype=np.float)
	xTest20 = np.column_stack((np.ones(TEST20_DATA.shape[0]), TEST20_DATA[:, 0:(TEST20_DATA.shape[1] - 1)]))
	yTest20 = TEST20_DATA[:, (TEST20_DATA.shape[1] - 1)]

	w20 = SGD(xTrain20, yTrain20, eta, Times)
	eOut = errRate(xTest20, yTest20, w20)
	t1 = time.time()
	print '========================================================='
	print 'Question 20:', eOut, 'with w', w20
	print '---------------------------------------------------------'
	print 'Q20 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()