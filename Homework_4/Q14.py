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
	TRAIN14_FILE = 'hw4_train.dat'
	TRAIN14_DATA = np.loadtxt(TRAIN14_FILE, dtype=np.float)
	xTrain14 = np.column_stack((np.ones(TRAIN14_DATA.shape[0]), TRAIN14_DATA[:, 0:(TRAIN14_DATA.shape[1] - 1)]))
	yTrain14 = TRAIN14_DATA[:, (TRAIN14_DATA.shape[1] - 1)]

	TEST14_FILE = 'hw4_test.dat'
	TEST14_DATA = np.loadtxt(TEST14_FILE, dtype=np.float)
	xTest14 = np.column_stack((np.ones(TEST14_DATA.shape[0]), TEST14_DATA[:, 0:(TEST14_DATA.shape[1] - 1)]))
	yTest14 = TEST14_DATA[:, (TEST14_DATA.shape[1] - 1)]

	lambPowList = []
	eInList  = []
	eOutList = []
	for lambPower in range(-10, 3):
		wREG = RLRV(xTrain14, yTrain14, math.pow(LAMB, lambPower))
		eIn	 = errRate(xTrain14, yTrain14, wREG)
		eOut = errRate(xTest14,  yTest14,  wREG)
		lambPowList.append(lambPower)
		eInList.append(eIn)
		eOutList.append(eOut)
	eInList  = np.array(eInList)
	minIndex = np.where(eInList == eInList.min())
	index    = minIndex[0].max()
	t1 = time.time()
	print '========================================================='
	# if len(minIndex[0]) > 1:
	# 	print 'Question 14:'
	# 	for index in minIndex[0]:
	# 		print 'log(lambda) is', lambPowList[index], 'Ein is', eInList[index], 'and Eout is', eOutList[index]
	# else:
	# 	index = minIndex[0][0]
	# 	print 'Question 14: log(lambda) is', lambPowList[index], 'Ein is', eInList[index], 'and Eout is', eOutList[index]
	print 'Question 14: log(lambda) is', lambPowList[index], 'Ein is', eInList[index], 'and Eout is', eOutList[index]
	print '---------------------------------------------------------'
	print 'Q14 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()