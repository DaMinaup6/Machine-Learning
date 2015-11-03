import math
import time
import numpy as np
import matplotlib.pyplot as plt

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

def plotHist(x, xLabel, yLabel, title, width, isFloat):
    plt.title(str(title))
    plt.xlabel(str(xLabel))
    plt.ylabel(str(yLabel))
    if isFloat: plt.hist(x)
    else:
        freq = np.bincount(x)
        freqIndex = np.nonzero(freq)[0]
        plt.bar(freqIndex, freq[freqIndex], width)
    plt.grid(True)
    plt.draw()
    plt.savefig(title)
    plt.close()

def main():
	repeat = 1000

	# errTot = 0.0
	# wLinTot = np.zeros(6)
	eOutList = []
	t0 = time.time()
	# for times in range(repeat):
	# 	(xArr, y15) = dataGeneration(1000, 2, 0.1)
	# 	x15 = np.column_stack((np.ones(xArr.shape[0]), featVec(xArr)))
	# 	wLin = np.dot(np.linalg.pinv(x15), y15)
	# 	wLinTot += wLin
	# 	errTot += errRate(x15, y15, wLin)
	# wLinTot /= repeat
	for times in range(repeat):
		(outArr, outY) = dataGeneration(1000, 2, 0.1)
		outX = np.column_stack((np.ones(outArr.shape[0]), featVec(outArr)))
		wLin = np.dot(np.linalg.pinv(outX), outY)
		eOutList.append(errRate(outX, outY, wLin))
	errOut = sum(eOutList) / float(repeat)
	plotHist(eOutList, "Eout Error Rate", "Frequency", "Q15", 0.01, True)
	t1 = time.time()
	print '========================================================='
	print 'Question 15:', errOut
	print '---------------------------------------------------------'
	print 'Q15 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()