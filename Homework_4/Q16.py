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
	TRAIN16_FILE = 'hw4_train_subtract.dat'
	TRAIN16_DATA = np.loadtxt(TRAIN16_FILE, dtype=np.float)
	xTrain16 = np.column_stack((np.ones(TRAIN16_DATA.shape[0]), TRAIN16_DATA[:, 0:(TRAIN16_DATA.shape[1] - 1)]))
	yTrain16 = TRAIN16_DATA[:, (TRAIN16_DATA.shape[1] - 1)]

	VALD16_FILE = 'hw4_validate.dat'
	VALD16_DATA = np.loadtxt(VALD16_FILE, dtype=np.float)
	xVald16 = np.column_stack((np.ones(VALD16_DATA.shape[0]), VALD16_DATA[:, 0:(VALD16_DATA.shape[1] - 1)]))
	yVald16 = VALD16_DATA[:, (VALD16_DATA.shape[1] - 1)]

	TEST16_FILE = 'hw4_test.dat'
	TEST16_DATA = np.loadtxt(TEST16_FILE, dtype=np.float)
	xTest16 = np.column_stack((np.ones(TEST16_DATA.shape[0]), TEST16_DATA[:, 0:(TEST16_DATA.shape[1] - 1)]))
	yTest16 = TEST16_DATA[:, (TEST16_DATA.shape[1] - 1)]

	lambList = []
	eInList  = []
	eValList = []
	eOutList = []
	for lambPower in range(-10, 3):
		wREG = RLRV(xTrain16, yTrain16, math.pow(LAMB, lambPower))
		eIn	 = errRate(xTrain16, yTrain16, wREG)
		eVal = errRate(xVald16,  yVald16,  wREG)
		eOut = errRate(xTest16,  yTest16,  wREG)
		lambList.append(lambPower)
		eInList.append(eIn)
		eValList.append(eVal)
		eOutList.append(eOut)
	eInList  = np.array(eInList)
	minIndex = np.where(eInList == eInList.min())
	t1 = time.time()
	print '========================================================='
	if len(minIndex[0]) > 1:
		print 'Question 16:'
		for index in minIndex[0]:
			print 'log(lambda) is', lambList[index], 'Etrain is', eInList[index], ', Eval is', eValList[index], 'and Eout is', eOutList[index]
	else:
		index = minIndex[0][0]
		print 'Question 16: log(lambda) is', lambList[index], 'Etrain is', eInList[index], ', Eval is', eValList[index], 'and Eout is', eOutList[index]
	print '---------------------------------------------------------'
	print 'Q16 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()