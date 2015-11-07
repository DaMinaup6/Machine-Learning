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

	t0 = time.time()
	TRAIN15_FILE = 'hw4_train.dat'
	TRAIN15_DATA = np.loadtxt(TRAIN15_FILE, dtype=np.float)
	xTrain15 = np.column_stack((np.ones(TRAIN15_DATA.shape[0]), TRAIN15_DATA[:, 0:(TRAIN15_DATA.shape[1] - 1)]))
	yTrain15 = TRAIN15_DATA[:, (TRAIN15_DATA.shape[1] - 1)]

	TEST15_FILE = 'hw4_test.dat'
	TEST15_DATA = np.loadtxt(TEST15_FILE, dtype=np.float)
	xTest15 = np.column_stack((np.ones(TEST15_DATA.shape[0]), TEST15_DATA[:, 0:(TEST15_DATA.shape[1] - 1)]))
	yTest15 = TEST15_DATA[:, (TEST15_DATA.shape[1] - 1)]

	lambList = []
	eInList  = []
	eOutList = []
	for lambPower in range(-10, 3):
		wREG = RLRV(xTrain15, yTrain15, math.pow(LAMB, lambPower))
		eIn	 = errRate(xTrain15, yTrain15, wREG)
		eOut = errRate(xTest15,  yTest15,  wREG)
		lambList.append(lambPower)
		eInList.append(eIn)
		eOutList.append(eOut)
	eOutList = np.array(eOutList)
	minIndex = np.where(eOutList == eOutList.min())
	t1 = time.time()
	print '========================================================='
	if len(minIndex[0]) > 1:
		print 'Question 15:'
		for index in minIndex[0]:
			print 'log(lambda) is', lambList[index], 'Ein is', eInList[index], 'and Eout is', eOutList[index]
	else:
		index = minIndex[0][0]
		print 'Question 15: log(lambda) is', lambList[index], 'Ein is', eInList[index], 'and Eout is', eOutList[index]
	print '---------------------------------------------------------'
	print 'Q15 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()