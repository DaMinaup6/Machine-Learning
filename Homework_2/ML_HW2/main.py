import DLA as dla
import numpy as np
import time

def main():
    # Question 17
    DATA_SIZE = 20
    NOISE = 0.2

    t0 = time.time()
    H = dla.HGenetation(DATA_SIZE)
    (eIn, eOut, eInList, eOutList) = dla.repeatDSA(H, DATA_SIZE, NOISE, 5000)
    dla.plotHist(eInList, "Error Rate", "Frequency", "Q17", 0.01, True)
    t1 = time.time()

    print "====================================="
    print "Q17: ", eIn
    print "-------------------------------------"
    print "Q17 costs ", t1 - t0, ' seconds'
    print "====================================="

    # Question 18
    t0 = time.time()
    H = dla.HGenetation(DATA_SIZE)
    (eIn, eOut, eInList, eOutList) = dla.repeatDSA(H, DATA_SIZE, NOISE, 5000)
    dla.plotHist(eOutList, "Error Rate", "Frequency", "Q18", 0.01, True)
    t1 = time.time()

    print "Q18: ", eOut
    print "-------------------------------------"
    print "Q18 costs ", t1 - t0, ' seconds'
    print "====================================="

    # Question 19
    TRAIN_FILE = "hw2_train.dat"
    TRAIN_DATA = np.loadtxt(TRAIN_FILE, dtype=np.float)

    t0 = time.time()
    x = TRAIN_DATA[:, 0:9]
    y = TRAIN_DATA[:, 9]
    H = dla.HGenetation(len(y))
    (eIn, h, data, eIndex) = dla.mulDSA(x, y, H)
    t1 = time.time()

    print "Q19: E_in: ", eIn, ", index: ", eIndex
    print "-------------------------------------"
    print "Q19 costs ", t1 - t0, ' seconds'
    print "====================================="

    # Question 20
    TEST_FILE = "hw2_test.dat"
    TEST_DATA = np.loadtxt(TEST_FILE, dtype=np.float)

    t0 = time.time()
    x = TRAIN_DATA[:, 0:9]
    y = TRAIN_DATA[:, 9]
    H = dla.HGenetation(len(y))
    (eIn, h, data, eIndex) = dla.mulDSA(x, y, H)
    (s, theta) = dla.sTheta(data, np.argmin(h), np.argmax(h))

    x = TEST_DATA[:, 0:9]
    y = TEST_DATA[:, 9]
    eOut = dla.errOut(s, theta, x, y)
    t1 = time.time()

    print "Q20: ", eOut
    print "-------------------------------------"
    print "Q20 costs ", t1 - t0, ' seconds'
    print "====================================="

if __name__ == '__main__':
    main()