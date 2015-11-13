import math
import time
import numpy as np
import matplotlib.pyplot as plt

def RLRV(x, y, lamb):
	XTX = np.dot(np.transpose(x), x)
	XTY = np.dot(np.transpose(x), y)
	return np.dot(np.linalg.inv(XTX + lamb * np.identity(XTX.shape[0])), XTY)

def errRate(x, y, w):
    yS = np.dot(x, w)
    yS[yS <= 0] = -1.0
    yS[yS > 0] = 1.0
    yErr = yS[yS != y]
    errCount = yErr.shape[0]
    return float(errCount) / len(y)

def main():
	LAMB = 11.26

	t0 = time.time()
	TRAIN13_FILE = 'hw4_train.dat'
	TRAIN13_DATA = np.loadtxt(TRAIN13_FILE, dtype=np.float)
	xTrain13 = np.column_stack((np.ones(TRAIN13_DATA.shape[0]), TRAIN13_DATA[:, 0:(TRAIN13_DATA.shape[1] - 1)]))
	yTrain13 = TRAIN13_DATA[:, (TRAIN13_DATA.shape[1] - 1)]

	TEST13_FILE = 'hw4_test.dat'
	TEST13_DATA = np.loadtxt(TEST13_FILE, dtype=np.float)
	xTest13 = np.column_stack((np.ones(TEST13_DATA.shape[0]), TEST13_DATA[:, 0:(TEST13_DATA.shape[1] - 1)]))
	yTest13 = TEST13_DATA[:, (TEST13_DATA.shape[1] - 1)]

	wREG = RLRV(xTrain13, yTrain13, LAMB)
	eIn	 = errRate(xTrain13, yTrain13, wREG)
	eOut = errRate(xTest13,  yTest13,  wREG)
	t1 = time.time()
	print '========================================================='
	print 'Question 13: Ein is', eIn, 'and Eout is', eOut
	print '---------------------------------------------------------'
	print 'Q13 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()