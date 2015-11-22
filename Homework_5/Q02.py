import math
import time
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

def phi(x):
    z = []
    for i in range(len(x)):
        z.append([math.pow(x[i][1], 2) - 2 * x[i][0] + 3, math.pow(x[i][0], 2) - 2 * x[i][1] - 3])
    return np.array(z)

def LinHardMarginSVM(x, y):
    Q_Mat = matrix(np.vstack((np.zeros(len(x[0]) + 1, dtype=np.float), np.column_stack((np.zeros(len(x[0]), dtype=np.float), np.identity(len(x[0]), dtype=np.float))))), tc='d')
    p_Mat = matrix(np.zeros(len(x[0]) + 1, dtype=np.float), tc='d')
    A_Mat = -1 * matrix(np.array([yI * xI for yI, xI in zip(y, np.column_stack((np.ones(len(x), dtype=np.float), x)))]), tc='d')
    c_Mat = -1 * matrix(np.ones(len(x), dtype=np.float), tc='d')
    return solvers.qp(Q_Mat, p_Mat, A_Mat, c_Mat)

def main():
    TRAIN02_FILE = 'ML_HW5_Q02.dat'

    TRAIN02_DATA = np.loadtxt(TRAIN02_FILE, dtype=np.float)
    z02 = phi(TRAIN02_DATA[:, 0:TRAIN02_DATA.shape[1] - 1])
    y02 = TRAIN02_DATA[:, (TRAIN02_DATA.shape[1] - 1)]

    t0 = time.time()
    u  = LinHardMarginSVM(z02, y02)['x']
    t1 = time.time()
    print '========================================================='
    print 'Question 02:', u
    print '---------------------------------------------------------'
    print 'Q02 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()