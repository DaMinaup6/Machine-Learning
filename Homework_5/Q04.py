import time
import numpy as np
import sympy as sp
import cvxopt as cvx

def kerFunc(x1, x2):
    return sp.expand((1 + np.dot(x1, x2)) ** 2)

def kerMatrix(x, y):
    kerMat = np.zeros((x.shape[0], y.shape[0]))
    for i in range(len(x)):
        for j in range(len(x)):
            kerMat[i, j] = y[i] * y[j] * kerFunc(x[i], x[j])
    return kerMat

def KerHardMarginSVM(x, y):
    Q_Mat = cvx.matrix(kerMatrix(x, y), tc='d')
    p_Mat = -1 * cvx.matrix(np.ones(len(x), dtype=np.float), tc='d')
    A_Mat = -1 * cvx.matrix(np.identity(len(y), dtype=np.float), tc='d')
    c_Mat = cvx.matrix(np.zeros(len(x), dtype=np.float), tc='d')
    G_Mat = cvx.matrix(np.transpose(y), tc='d')
    h_Mat = cvx.matrix(0, tc='d')
    return cvx.solvers.qp(Q_Mat, p_Mat, A_Mat, c_Mat, G_Mat, h_Mat)

def main():
    TRAIN04_FILE = 'ML_HW5_Q02.dat'

    TRAIN04_DATA = np.loadtxt(TRAIN04_FILE, dtype=np.float)
    x04 = TRAIN04_DATA[:, 0:TRAIN04_DATA.shape[1] - 1]
    y04 = TRAIN04_DATA[:, TRAIN04_DATA.shape[1] - 1:TRAIN04_DATA.shape[1]]

    t0  = time.time()
    alp = KerHardMarginSVM(x04, y04)['x']
    sv  = np.argmax(alp)
    w   = sum([alpha * y * kerFunc(x, np.array([sp.Symbol('x1'), sp.Symbol('x2')])) for alpha, y, x in zip(alp, y04, x04)])
    b   = y04[sv] - sum([alpha * y * kerFunc(x, x04[sv]) for alpha, y, x in zip(alp, y04, x04)])
    t1  = time.time()
    print '========================================================='
    print 'Question 04:', w + b
    print '---------------------------------------------------------'
    print 'Q04 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()