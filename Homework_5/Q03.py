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

def kerFunc(x1, x2):
    return math.pow(1 + np.dot(x1, x2), 2)

def kerMatrix(x, y):
    kerMat = np.zeros((x.shape[0], y.shape[0]))
    for i in range(len(x)):
        for j in range(len(x)):
            kerMat[i, j] = y[i] * y[j] * kerFunc(x[i], x[j])
    return kerMat

def KerHardMarginSVM(x, y):
    Q_Mat = matrix(kerMatrix(x, y), tc='d')
    p_Mat = -1 * matrix(np.ones(len(x), dtype=np.float), tc='d')
    A_Mat = -1 * matrix(np.identity(len(y), dtype=np.float), tc='d')
    c_Mat = matrix(np.zeros(len(x), dtype=np.float), tc='d')
    G_Mat = matrix(np.transpose(y), tc='d')
    h_Mat = matrix(0, tc='d')
    return solvers.qp(Q_Mat, p_Mat, A_Mat, c_Mat, G_Mat, h_Mat)

# def latexMat(mat, newLine):
#     string = ''
#     count  = 0
#     for i in mat:
#         string += str(i)
#         count += 1
#         if count % newLine == 0:
#             string += '\\\\'
#             count   = 0
#         else:
#             string += '&'
#     string = string[:-1]
#     return string[:-1]

def main():
    TRAIN03_FILE = 'ML_HW5_Q02.dat'

    TRAIN03_DATA = np.loadtxt(TRAIN03_FILE, dtype=np.float)
    x03 = TRAIN03_DATA[:, 0:TRAIN03_DATA.shape[1] - 1]
    y03 = TRAIN03_DATA[:, TRAIN03_DATA.shape[1] - 1:TRAIN03_DATA.shape[1]]

    t0  = time.time()
    alp = KerHardMarginSVM(x03, y03)['x']
    t1  = time.time()
    print '========================================================='
    print 'Question 03:', alp
    print '---------------------------------------------------------'
    print 'Q03 costs', t1 - t0, 'seconds'
    print '========================================================='

if __name__ == '__main__':
    main()