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
	LAMB = 10.0
	SPLIT = 40

	t0 = time.time()

	TRAIN20_FILE = 'hw4_train.dat'
	TRAIN20_DATA = np.loadtxt(TRAIN20_FILE, dtype=np.float)
	xTrain20 = np.column_stack((np.ones(TRAIN20_DATA.shape[0]), TRAIN20_DATA[:, 0:(TRAIN20_DATA.shape[1] - 1)]))
	yTrain20 = TRAIN20_DATA[:, (TRAIN20_DATA.shape[1] - 1)]

	TEST20_FILE = 'hw4_test.dat'
	TEST20_DATA = np.loadtxt(TEST20_FILE, dtype=np.float)
	xTest20 = np.column_stack((np.ones(TEST20_DATA.shape[0]), TEST20_DATA[:, 0:(TEST20_DATA.shape[1] - 1)]))
	yTest20 = TEST20_DATA[:, (TEST20_DATA.shape[1] - 1)]

	lambPowList = []
	eCvList     = []
	for lambPower in range(-10, 3):
		eCv = vFoldErr(xTrain20, yTrain20, math.pow(LAMB, lambPower), SPLIT)
		lambPowList.append(lambPower)
		eCvList.append(eCv)
	eCvList  = np.array(eCvList)
	minIndex = np.where(eCvList == eCvList.min())
	index    = minIndex[0].max()

	wREG = RLRV(xTrain20, yTrain20, math.pow(LAMB, lambPowList[index]))
	eIn  = errRate(xTrain20, yTrain20, wREG)
	eOut = errRate(xTest20,  yTest20,  wREG)

	t1 = time.time()
	print '========================================================='
	print 'Question 20: Ein is', eIn, 'Eout is', eOut
	print '---------------------------------------------------------'
	print 'Q20 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()