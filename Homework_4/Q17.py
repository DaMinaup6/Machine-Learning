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
	TRAIN17_FILE = 'hw4_train.dat'
	TRAIN17_DATA = np.loadtxt(TRAIN17_FILE, dtype=np.float)
	xTrain17 = np.column_stack((np.ones(SPLIT), TRAIN17_DATA[0:SPLIT, 0:(TRAIN17_DATA.shape[1] - 1)]))
	yTrain17 = TRAIN17_DATA[0:SPLIT, (TRAIN17_DATA.shape[1] - 1)]

	VALD17_FILE = 'hw4_train.dat'
	VALD17_DATA = np.loadtxt(VALD17_FILE, dtype=np.float)
	xVald17 = np.column_stack((np.ones(TRAIN17_DATA.shape[0] - SPLIT), VALD17_DATA[SPLIT:, 0:(VALD17_DATA.shape[1] - 1)]))
	yVald17 = VALD17_DATA[SPLIT:, (VALD17_DATA.shape[1] - 1)]

	TEST17_FILE = 'hw4_test.dat'
	TEST17_DATA = np.loadtxt(TEST17_FILE, dtype=np.float)
	xTest17 = np.column_stack((np.ones(TEST17_DATA.shape[0]), TEST17_DATA[:, 0:(TEST17_DATA.shape[1] - 1)]))
	yTest17 = TEST17_DATA[:, (TEST17_DATA.shape[1] - 1)]

	lambPowList = []
	eInList  = []
	eValList = []
	eOutList = []
	for lambPower in range(-10, 3):
		wREG = RLRV(xTrain17, yTrain17, math.pow(LAMB, lambPower))
		eIn	 = errRate(xTrain17, yTrain17, wREG)
		eVal = errRate(xVald17,  yVald17,  wREG)
		eOut = errRate(xTest17,  yTest17,  wREG)
		lambPowList.append(lambPower)
		eInList.append(eIn)
		eValList.append(eVal)
		eOutList.append(eOut)
	eValList = np.array(eValList)
	minIndex = np.where(eValList == eValList.min())
	index    = minIndex[0].max()
	t1 = time.time()
	print '========================================================='
	# if len(minIndex[0]) > 1:
	# 	print 'Question 17:'
	# 	for index in minIndex[0]:
	# 		print 'log(lambda) is', lambPowList[index], 'Etrain is', eInList[index], ', Eval is', eValList[index], 'and Eout is', eOutList[index]
	# else:
	# 	index = minIndex[0][0]
	# 	print 'Question 17: log(lambda) is', lambPowList[index], 'Etrain is', eInList[index], ', Eval is', eValList[index], 'and Eout is', eOutList[index]
	print 'Question 17: log(lambda) is', lambPowList[index], 'Etrain is', eInList[index], ', Eval is', eValList[index], 'and Eout is', eOutList[index]
	print '---------------------------------------------------------'
	print 'Q17 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()