import myPLA as pla
import numpy as np
import time

def main():
    TRAIN15_FILE = "ntumlone%2Fhw1%2Fhw1_15_train.dat"
    TRAIN18_FILE = "ntumlone%2Fhw1%2Fhw1_18_train.dat"
    TEST18_FILE = "ntumlone%2Fhw1%2Fhw1_18_test.dat"

    TRAIN15_DATA = np.loadtxt(TRAIN15_FILE, dtype=np.float)
    x15 = np.column_stack((np.ones(TRAIN15_DATA.shape[0]), TRAIN15_DATA[:, 0:(TRAIN15_DATA.shape[1] - 1)]))
    y15 = TRAIN15_DATA[:, (TRAIN15_DATA.shape[1] - 1)]

    TRAIN18_DATA = np.loadtxt(TRAIN18_FILE, dtype=np.float)
    x18 = np.column_stack((np.ones(TRAIN18_DATA.shape[0]), TRAIN18_DATA[:, 0:(TRAIN18_DATA.shape[1] - 1)]))
    y18 = TRAIN18_DATA[:, (TRAIN18_DATA.shape[1] - 1)]

    TEST18_DATA = np.loadtxt(TEST18_FILE, dtype=np.float)
    a = np.column_stack((np.ones(TEST18_DATA.shape[0]), TEST18_DATA[:, 0:(TEST18_DATA.shape[1] - 1)]))
    b = TEST18_DATA[:, (TEST18_DATA.shape[1] - 1)]

    print "====================================="
    t0 = time.time()
    (w15, update, index) = pla.PLA(x15, y15, list(xrange(len(x15))), 1.0)
    t1 = time.time()
    print "Question 15: update ", int(update), ", index ", int(index)
    print "-------------------------------------"
    print "Q15 costs ", t1 - t0, ' seconds'
    print "====================================="

    t2 = time.time()
    (aveUpdate16, updateHistQ16, Q16AveTime) = pla.randomSeedPLA(x15, y15, 1.0, 2000)
    pla.plotHist(updateHistQ16, "Updates", "Frequency", "Q16", 1.0, False)
    t3 = time.time()
    print "Question 16: ", aveUpdate16
    print "-------------------------------------"
    print "Q16 costs ", t3 - t2, ' seconds'
    print "====================================="

    t4 = time.time()
    (aveUpdate17, updateHistQ17, Q17AveTime) = pla.randomSeedPLA(x15, y15, 0.5, 2000)
    pla.plotHist(updateHistQ17, "Updates", "Frequency", "Q17", 1.0, False)
    t5 = time.time()
    print "Question 17: ", aveUpdate17
    print "-------------------------------------"
    print "Q17 costs ", t5 - t4, ' seconds'
    print "====================================="

    t6 = time.time()
    (aveErr18, errHistQ18) = pla.randomSeedPoc(x18, y18, a, b, 1.0, 50, 2000, True)
    pla.plotHist(errHistQ18, "Error Rate", "Frequency", "Q18", 0.01, True)
    t7 = time.time()
    print "Question 18: ", aveErr18
    print "-------------------------------------"
    print "Q18 costs ", t7 - t6, ' seconds'
    print "====================================="

    t8 = time.time()
    (aveErr19, errHistQ19) = pla.randomSeedPoc(x18, y18, a, b, 1.0, 50, 2000, False)
    pla.plotHist(errHistQ19, "Error Rate", "Frequency", "Q19", 0.01, True)
    t9 = time.time()
    print "Question 19: ", aveErr19
    print "-------------------------------------"
    print "Q19 costs ", t9 - t8, ' seconds'
    print "====================================="

    t10 = time.time()
    (aveErr20, errHistQ20) = pla.randomSeedPoc(x18, y18, a, b, 1.0, 100, 2000, True)
    pla.plotHist(errHistQ20, "Error Rate", "Frequency", "Q20", 0.01, True)
    t11 = time.time()
    print "Question 20: ", aveErr20
    print "-------------------------------------"
    print "Q20 costs ", t11 - t10, ' seconds'
    print "====================================="

    # t12 = time.time()
    # (aveFactor21, factorHistQ21) = pla.PLATime(x15, y15, 1.0, 5, 2000, 1.0, 20.0)
    # pla.plotHist(factorHistQ21, "Q21/Q15", "Frequency", "Q21", 0.01, True)
    # t13 = time.time()
    # print "Question 21: ", aveFactor21
    # print "-------------------------------------"
    # print "Q21 costs ", t13 - t12, ' seconds'
    # print "====================================="

if __name__ == '__main__':
    main()