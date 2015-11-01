import math
import time
import numpy as np
from numpy.linalg import inv

def f(u, v):
	return np.exp(u) + np.exp(2 * v) + np.exp(u * v) + math.pow(u, 2) - 2 * u * v + 2 * math.pow(v, 2) - 3 * u - 2 * v

def gradien(u, v):
	fu = np.exp(u) + v * np.exp(u * v) + 2 * u - 2 * v - 3
	fv = 2 * np.exp(2 * v) + u * np.exp(u * v) - 2 * u + 4 * v - 2
	return np.array([[fu], [fv]])

def Hessian(u, v):
	fuu = np.exp(u) + math.pow(v, 2) * np.exp(u * v) + 2
	fuv = (u * v + 1) * np.exp(u * v) - 2
	fvu = (u * v + 1) * np.exp(u * v) - 2
	fvv = 4 * np.exp(2 * v) + math.pow(u, 2) * np.exp(u * v) + 4
	return np.array([[fuu, fuv], [fvu, fvv]])

def GDA(u0, v0, eta, update):
	times = 0
	histList = []
	while times < update:
		upTerm = np.dot(-inv(Hessian(u0, v0)), gradien(u0, v0))
		oMat = np.array([[u0], [v0]])
		nMat = oMat + upTerm
		u0 = nMat[0][0]
		v0 = nMat[1][0]
		histList.append((u0, v0, f(u0, v0)))
		times += 1
	return histList

def main():
	u0 = 0
	v0 = 0
	eta = 0.01
	update = 5

	t0 = time.time()
	histList = GDA(u0, v0, eta, update)
	t1 = time.time()

	times = 0
	while times < update:
		(u, v, val) = histList[times]
		times += 1
	print '========================================================='
	print 'Question 10: E(u,v) =', val, 'after', update, 'times updates, (u,v) = (' + str(u) + ', ' + str(v) + ').'
	print '---------------------------------------------------------'
	print 'Q10 costs', t1 - t0, 'seconds'
	print '========================================================='

if __name__ == '__main__':
    main()