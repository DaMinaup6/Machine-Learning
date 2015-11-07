import math
import time
import numpy as np

def RLRV(x, y, lamb):
	XTX = np.dot(np.transpose(x), x)
	XTY = np.dot(np.transpose(x), y)
	return np.dot(np.linalg.inv(XTX + lamb * np.identity(XTX.shape[0])), XTY)

def vFoldErr(x, y, lamb, split):
	vTime = len(y) / split
	times = 0
	errTot = 0.0
	while times < vTime:
		xV = np.delete(x, np.s_[(split * times):(split * (times + 1))], 0)
		yV = np.delete(y, np.s_[(split * times):(split * (times + 1))], 0)
		wV = RLRV(xV, yV, lamb)

		xVal = x[(split * times):(split * (times + 1))]
		yVal = y[(split * times):(split * (times + 1))]
		errTot += errRate(xVal, yVal, wV)
		times += 1
	if split * vTime < len(y):
		xV = np.delete(x, np.s_[(split * vTime):], 0)
		yV = np.delete(y, np.s_[(split * vTime):], 0)
		wV = RLRV(xV, yV, lamb)

		xVal = x[(split * vTime):]
		yVal = y[(split * vTime):]
		errTot += errRate(xVal, yVal, wV)
		vTime += 1
	return errTot / vTime

def errRate(x, y, w):
    yS = np.dot(x, w)
    yS[yS <= 0] = -1.0
    yS[yS > 0] = 1.0
    yErr = yS[yS != y]
    errCount = yErr.shape[0]
    return float(errCount) / len(y)

def main():
	LAMB = 10
	SPLIT = 40

	t0 = time.time()

	TRAIN19_FILE = 'hw4_train.dat'
	TRAIN19_DATA = np.loadtxt(TRAIN19_FILE, dtype=np.float)
	xTrain19 = np.column_stack((np.ones(TRAIN19_DATA.shape[0]), TRAIN19_DATA[:, 0:(TRAIN19_DATA.shape[1] - 1)]))
	yTrain19 = TRAIN19_DATA[:, (TRAIN19_DATA.shape[1] - 1)]

	TEST19_FILE = 'hw4_test.dat'
	TEST19_DATA = np.loadtxt(TEST19_FILE, dtype=np.float)
	xTest19 = np.column_stack((np.ones(TEST19_DATA.shape[0]), TEST19_DATA[:, 0:(TEST19_DATA.shape[1] - 1)]))
	yTest19 = TEST19_DATA[:, (TEST19_DATA.shape[1] - 1)]

	lambPowList = []
	eCvList     = []
	for lambPower in range(-10, 3):
		eCv = vFoldErr(xTrain19, yTrain19, math.pow(LAMB, lambPower), SPLIT)
		lambPowList.append(lambPower)
		eCvList.append(eCv)
	eCvList  = np.array(eCvList)
	minIndex = np.where(eCvList == eCvList.min())
	index    = minIndex[0].max()

	t1 = time.time()
	print '========================================================='
	print 'Question 19: log(lambda) is', lambPowList[index], 'Ecv is', eCvList[index]
	print '---------------------------------------------------------'
	print 'Q19 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()