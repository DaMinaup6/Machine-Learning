import numpy as np
import random as rd
import math
import time

def sign(x):
	if x >= 0:
		return +1.0
	else:
		return -1.0

def HGenetation(N):
    H = np.ones((N, N), dtype=np.float)
    index = 1
    for i in range(1, N):
        for j in range(0, index):
            H[j][i] = -1.0
        index += 1
    return np.concatenate((H, -H))

def mulDSA(x, y, H):
    einErr = []
    for xI in range(len(x[0])):
        xArr = x[:, xI]
        data = np.column_stack((xArr, y))
        data = np.array(sorted(data, key=lambda data: data[0]), dtype=np.float)
        (hArr, err) = errIn(data[:, 1], H)
        einErr.append(err)
    return (min(einErr), np.argmin(einErr))

def errIn(y, H):
    result = []
    for i in range(len(H)):
        yErr = H[i][H[i] != y]
        result.append(float(yErr.shape[0]))
    minInd = np.argmin(result)
    return (H[minInd], result[minInd] / len(y))

def main():
    t0 = time.time()
    TRAIN_FILE = "hw2_train.dat"
    TRAIN_DATA = np.loadtxt(TRAIN_FILE, dtype=np.float)
    x = TRAIN_DATA[:, 0:9]
    y = TRAIN_DATA[:, 9]
    H = HGenetation(len(y))
    (eIn, eIndex) = mulDSA(x, y, H)
    t1 = time.time()
    print "====================================="
    print "Q19: E_in: ", eIn, ", index: ", eIndex
    print "-------------------------------------"
    print "Q19 costs ", t1 - t0, ' seconds'
    print "====================================="

if __name__ == '__main__':
    main()