cimport cython
cimport numpy as np
import numpy as np
import time
import constants as ctn

def SVDResult():
    cdef int i, j, k, movie, user, nFeat, nUser, nTrain, nMov
    cdef double globalAvg, a, temp
    nFeat = ctn.SVDNumFeat()
    nUser = ctn.userNumber()
    nMov = ctn.movieNumber()
    nTrain = ctn.trainSize()
    cdef np.ndarray[double, ndim = 1] userFeat = np.fromfile(ctn.SVDUserFeatPath(), 'float')
    cdef np.ndarray[double, ndim = 1] movFeat = np.fromfile(ctn.SVDMovieFeatPath(), 'float')
    cdef np.ndarray[uint, ndim = 1] toRate = np.fromfile(ctn.qualPath(), 'uint32')
    cdef np.ndarray[double, ndim = 1] rating = np.empty(ctn.qualSize(), 'float')
    cdef np.ndarray[double, ndim = 1] movieAvg = np.empty((nMov), 'float')
    cdef np.ndarray[double, ndim = 1] userAvg = np.empty((nUser), 'float')
    cdef np.ndarray[double, ndim = 1] userOffset = np.empty((nUser), 'float')
    movieAvg = np.loadtxt(ctn.movieSmthAvgPath())
    userAvg = np.loadtxt(ctn.userSmthAvgPath())
    userOffset = np.loadtxt(ctn.userOffsetPath())
    globalAvg = ctn.globalAvg()
    
    start_time = time.time()
    for i in range(ctn.qualSize()):
        user = toRate[3*i] - 1
        movie = toRate[3*i+1] - 1
        a = movieAvg[movie] + userOffset[user]
        for j in range(nFeat):
            a += (movFeat[movie*nFeat + j] * userFeat[user*nFeat + j])
            if (a > 5):
                a = 5
            elif (a < 1):
                a = 1
        rating[i] = a
    print 'Computing ratings took: ', time.time() - start_time, "seconds"
    
    np.savetxt(ctn.SVDResultPath(), rating)
        
