import math
import time
import numpy as np

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
	LAMB = 10
	SPLIT = 120

	t0 = time.time()

	TRAIN18_FILE = 'hw4_train.dat'
	TRAIN18_DATA = np.loadtxt(TRAIN18_FILE, dtype=np.float)
	xTrain18 = np.column_stack((np.ones(SPLIT), TRAIN18_DATA[0:SPLIT, 0:(TRAIN18_DATA.shape[1] - 1)]))
	yTrain18 = TRAIN18_DATA[0:SPLIT, (TRAIN18_DATA.shape[1] - 1)]

	VALD18_FILE = 'hw4_train.dat'
	VALD18_DATA = np.loadtxt(VALD18_FILE, dtype=np.float)
	xVald18 = np.column_stack((np.ones(TRAIN18_DATA.shape[0] - SPLIT), VALD18_DATA[SPLIT:, 0:(VALD18_DATA.shape[1] - 1)]))
	yVald18 = VALD18_DATA[SPLIT:, (VALD18_DATA.shape[1] - 1)]

	TEST18_FILE = 'hw4_test.dat'
	TEST18_DATA = np.loadtxt(TEST18_FILE, dtype=np.float)
	xTest18 = np.column_stack((np.ones(TEST18_DATA.shape[0]), TEST18_DATA[:, 0:(TEST18_DATA.shape[1] - 1)]))
	yTest18 = TEST18_DATA[:, (TEST18_DATA.shape[1] - 1)]

	lambPowList = []
	eInList  = []
	eValList = []
	eOutList = []
	for lambPower in range(-10, 3):
		wREG = RLRV(xTrain18, yTrain18, math.pow(LAMB, lambPower))
		eIn	 = errRate(xTrain18, yTrain18, wREG)
		eVal = errRate(xVald18,  yVald18,  wREG)
		eOut = errRate(xTest18,  yTest18,  wREG)
		lambPowList.append(lambPower)
		eInList.append(eIn)
		eValList.append(eVal)
		eOutList.append(eOut)
	eValList = np.array(eValList)
	minIndex = np.where(eValList == eValList.min())
	index    = minIndex[0].max()

	TRAIN18_FILE = 'hw4_train.dat'
	TRAIN18_DATA = np.loadtxt(TRAIN18_FILE, dtype=np.float)
	xTrain18 = np.column_stack((np.ones(TRAIN18_DATA.shape[0]), TRAIN18_DATA[:, 0:(TRAIN18_DATA.shape[1] - 1)]))
	yTrain18 = TRAIN18_DATA[:, (TRAIN18_DATA.shape[1] - 1)]

	wREG = RLRV(xTrain18, yTrain18, math.pow(LAMB, lambPowList[index]))
	eIn	 = errRate(xTrain18, yTrain18, wREG)
	eOut = errRate(xTest18,  yTest18,  wREG)

	t1 = time.time()
	print '========================================================='
	print 'Question 18: Ein is', eIn, 'and Eout is', eOut
	print '---------------------------------------------------------'
	print 'Q18 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()