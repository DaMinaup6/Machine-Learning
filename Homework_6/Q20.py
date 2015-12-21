import time
import numpy as np

def sign(x):
	if x >= 0:
		return +1.0
	else:
		return -1.0

def gaussianKer(x1, x2, gamma):
    return np.exp(-gamma * (np.linalg.norm(x1 - x2) ** 2))

def KmatrixList(x, gammaList):
    kMatList = []
    for gamma in gammaList:
        kerMat = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                kerMat[i, j] = gaussianKer(x[i], x[j], gamma)
        kMatList.append((kerMat, gamma))
    return kMatList

def betaList(lambList, kMatList, y):
    N = len(y)
    I = np.identity(N)
    bList = []
    for lamb in lambList:
        for K in kMatList:
            beta = np.dot(np.linalg.inv(lamb * I + K[0]), y)
            bList.append((beta, lamb, K[1]))
    return bList

def errOutList(betaList, xTra, xTes, y):
    eOutList = []
    for B in betaList:
        yS    = []
        beta  = B[0]
        gamma = B[2]
        for i in range(xTes.shape[0]):
            summation = 0.0
            for j in range(xTra.shape[0]):
                summation += beta[j] * gaussianKer(xTra[j], xTes[i], gamma)
            yS.append(sign(summation))
        eOutList.append(np.mean((yS != y).astype(int)))
    return eOutList


def main():
    SPLIT = 400

    t0 = time.time()

    DATA20_FILE = 'hw2_lssvm_all.dat'
    DATA20_DATA = np.loadtxt(DATA20_FILE, dtype=np.float)
    xTra20 = DATA20_DATA[0:SPLIT, 0:(DATA20_DATA.shape[1] - 1)]
    yTra20 = DATA20_DATA[0:SPLIT,   (DATA20_DATA.shape[1] - 1)]
    xTes20 = DATA20_DATA[SPLIT: , 0:(DATA20_DATA.shape[1] - 1)]
    yTes20 = DATA20_DATA[SPLIT: ,   (DATA20_DATA.shape[1] - 1)]

    gammList = [32, 2, 0.125]
    lambList = [0.001, 1, 1000]

    kMatList = KmatrixList(xTra20, gammList)
    bVecList = betaList(lambList, kMatList, yTra20)
    eOutList = errOutList(bVecList, xTra20, xTes20, yTes20)
    eMinIdx  = np.argmin(eOutList)

    eOutMin  = eOutList[eMinIdx]
    eMinLamb = bVecList[eMinIdx][1]
    eMinGamm = bVecList[eMinIdx][2]

    t1 = time.time()

    print "============================================"
    print "Q20:", "min Eout(g) =", eOutMin, ", with lambda =", eMinLamb, ', gamma =', eMinGamm
    print "--------------------------------------------"
    print "Q20 costs", t1 - t0, 'seconds'
    print "============================================"

if __name__ == '__main__':
    main()