cimport cython
cimport numpy as np
import numpy as np
import time
import constants as ctn
from math import log, floor

def SVDResult():
    cdef int i, j, k, movie, user, freq, nFeat, nUser, nTrain, nMov
    cdef double globalAvg, a, temp
    nFeat = ctn.SVDNumFeat()
    nUser = ctn.userNumber()
    nMov = ctn.movieNumber()
    nTrain = ctn.trainSize()
    cdef np.ndarray[double, ndim = 1] userFeat = np.fromfile(ctn.SVDUserFeatPath(), 'float')
    cdef np.ndarray[double, ndim = 1] movFeat = np.fromfile(ctn.SVDMovieFeatPath(), 'float')
    cdef np.ndarray[uint, ndim = 1] toRate = np.fromfile(ctn.qualPath(), 'uint32')
    cdef np.ndarray[double, ndim = 1] rating = np.empty(ctn.qualSize(), 'float')
    cdef np.ndarray[double, ndim = 1] movieAvg = np.empty(nMov, 'float')
    cdef np.ndarray[double, ndim = 1] userAvg = np.empty(nUser, 'float')
    cdef np.ndarray[double, ndim = 1] userOffset = np.empty(nUser, 'float')
    cdef np.ndarray[double, ndim = 2] freqOffset = np.empty([nMov, 12], 'float')
    cdef np.ndarray[uint, ndim=1] userdata = np.fromfile(ctn.userPath(), 'uint32')
    cdef np.ndarray[ushort, ndim=1] userRates = np.empty(nUser, 'uint16')
    movieAvg = np.loadtxt(ctn.movieSmthAvgPath())
    userAvg = np.loadtxt(ctn.userSmthAvgPath())
    userOffset = np.loadtxt(ctn.userOffsetPath())
    freqOffset = np.loadtxt(ctn.freqOffsetPath())
    globalAvg = ctn.globalAvg()
    
    start_time = time.time()
    cdef double base = ctn.freqLogBase()

    for i in range(nTrain):
        user = userdata[i] - 1
        userRates[user] += 1

    for i in range(ctn.qualSize()):
        user = toRate[3*i] - 1
        movie = toRate[3*i+1] - 1
        freq = floor(log(userRates[user], base))
        a = movieAvg[movie] + userOffset[user] + freqOffset[movie, freq]
        for j in range(nFeat):
            a += (movFeat[movie*nFeat + j] * userFeat[user*nFeat + j])
            if (a > 5):
                a = 5
            elif (a < 1):
                a = 1
        rating[i] = a
    print 'Computing ratings took: ', time.time() - start_time, "seconds"
    
    np.savetxt(ctn.SVDResultPath(), rating)
        
