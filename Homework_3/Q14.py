import math
import time
import numpy as np

def sign(data, const):
	if (math.pow(data[0], 2) + math.pow(data[1], 2) + const) > 0:
		return 1.0
	else:
		return -1.0

def featVec(xArr):
	fVec = []
	for x in xArr:
		fVec.append([x[0], x[1], x[0] * x[1], math.pow(x[0], 2), math.pow(x[1], 2)])
	return np.array(fVec)

def errRate(x, y, w):
    yS = np.dot(x, w)
    yS[yS <= 0] = -1.0
    yS[yS > 0] = 1.0
    yErr = yS[yS != y]
    errCount = yErr.shape[0]
    return float(errCount) / len(y)

def dataGeneration(dataSize, dataDimension, noise):
	yArray = []
	dArray = np.array(2 * np.random.random_sample((dataSize, dataDimension)) - 1)
	for data in dArray:
		y = sign(data, -0.6)
		flip = np.random.random()
		if flip < noise:
			y = -y
		yArray.append(y)
	return (dArray, np.array(yArray))

def main():
	repeat = 1000

	errTot = 0.0
	wLinTot = np.zeros(6)
	t0 = time.time()
	for times in range(repeat):
		(xArr, y14) = dataGeneration(1000, 2, 0.1)
		x14 = np.column_stack((np.ones(xArr.shape[0]), featVec(xArr)))
		wLin = np.dot(np.linalg.pinv(x14), y14)
		wLinTot += wLin
		errTot += errRate(x14, y14, wLin)
	wLinTot /= repeat
	t1 = time.time()
	print '========================================================='
	print 'Question 14:', wLin
	print '---------------------------------------------------------'
	print 'Q14 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()