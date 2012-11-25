cimport cython
cimport numpy as np
import numpy as np
import time
from math import sqrt, log, floor
import constants as ctn

@cython.boundscheck(False)
@cython.wraparound(False)

def svd():
    cdef int i, j, k, p, user, movie, freq, nIter, nFeat, nUser, nMov, nTrain
    cdef double lrate, pred, delta, reg, u, m, temp
    lrate = ctn.SVDLearnRate()
    reg = lrate * ctn.SVDRegRatio()
    nIter = ctn.SVDNumEpoch()
    nFeat = ctn.SVDNumFeat()
    nUser = ctn.userNumber()
    nMov = ctn.movieNumber()
    nTrain = ctn.trainSize()
    cdef np.ndarray[double, ndim=2] userFeature = np.zeros([nUser, nFeat], 'float') + 0.1
    cdef np.ndarray[double, ndim=2] movieFeature = np.zeros([nMov, nFeat], 'float') + 0.1
    cdef np.ndarray[uint, ndim=1] userdata = np.fromfile(ctn.userPath(), 'uint32')
    cdef np.ndarray[uint, ndim=1] moviedata = np.fromfile(ctn.moviePath(), 'uint32')
    cdef np.ndarray[char, ndim=1] ratingdata = np.fromfile(ctn.ratingPath(), 'int8')
    cdef np.ndarray[double, ndim = 1] movie_a = np.fromfile(ctn.movieAvgPath(), 'float')
    cdef np.ndarray[double, ndim = 1] user_a = np.fromfile(ctn.userAvgPath(), 'float')
    cdef np.ndarray[float, ndim = 1] pred_archive = np.empty(nTrain, 'float32')
    
    # Calculate a good baseline
    start_time = time.time()
    cdef double globalTotal = 0
    cdef np.ndarray[ushort, ndim=1] userRates = np.empty(nUser, 'uint16')
    cdef np.ndarray[ushort, ndim=1] movieRates = np.empty(nMov, 'uint16')
    cdef np.ndarray[float, ndim=1] movieAvg = np.empty(nMov, 'float32')
    cdef np.ndarray[float, ndim=1] userAvg = np.empty(nUser, 'float32')
    cdef np.ndarray[float, ndim=1] userOffset = np.empty(nUser, 'float32')
    
    cdef double globalAvg = 0
        
    cdef int w = 25    
    for i in range(nTrain):
        user = userdata[i] - 1
        movie = moviedata[i] - 1
        globalTotal += ratingdata[i]
        userRates[user] += 1
        movieRates[movie] += 1
        
    globalAvg = globalTotal / nTrain

    # Compute the smoothed movie_average
    cdef double totalRating = 0
    for i in range(nMov):
        totalRating = movie_a[i] * movieRates[i]
        movieAvg[i] = (globalAvg * w + totalRating) / (w + movieRates[i])
    for i in range(nUser):
        totalRating = user_a[i] * userRates[i]
        userAvg[i] = (globalAvg * w + totalRating) / (w + userRates[i])
        
    # Compute the user offset
    for i in range(nTrain):
        user = userdata[i] - 1
        movie = moviedata[i] - 1
        userOffset[user] += ratingdata[i] - movie_a[movie]
    for i in range(nUser):
        userOffset[i] = userOffset[i] / (userRates[i] + (w * 1.0))

    print "Computing baseline took: ", time.time() - start_time, "seconds"
    
    np.savetxt(ctn.movieSmthAvgPath(), movieAvg)
    np.savetxt(ctn.userSmthAvgPath(), userAvg)
    np.savetxt(ctn.userOffsetPath(), userOffset)
    print "Global average rating: ", globalAvg
    
    start_time = time.time()
    cdef np.ndarray[double, ndim=1] movief = np.empty(nMov, 'float')
    cdef np.ndarray[double, ndim=1] userf = np.empty(nUser, 'float')
    
    for i in range(nTrain):
        user = userdata[i] - 1
        movie = moviedata[i] - 1
        pred_archive[i] = movieAvg[movie] + userOffset[user]
        
    for j in range(nFeat):
        for i in range(nUser):
            userf[i] = userFeature[i, j]
        for i in range(nMov):   
            movief[i] = movieFeature[i, j]
        for k in range(nIter):
            for i in range(nTrain):
                user = userdata[i] - 1
                movie = moviedata[i] - 1
                u = userf[user]
                m = movief[movie]           
                temp = pred_archive[i] + u * m
                delta = ratingdata[i] - temp
                err = lrate * delta
                userf[user] = u + (err * m - reg * u) 
                movief[movie] = m + (err * u - reg * m)
        for i in range(nUser):
            userFeature[i, j] = userf[i]
        for i in range(nMov):
            movieFeature[i, j] = movief[i]
        for i in range(nTrain):
            user = userdata[i] - 1
            movie = moviedata[i] - 1
            pred_archive[i] += userf[user] * movief[movie]
        print "Feature ", j, ": Done"
    print "SVD took: ", time.time() - start_time, "seconds"

    # Compute the frequency offset
    start_time = time.time()
    cdef double base = ctn.freqLogBase()
    cdef int maxUserRates = np.amax(userRates)
    cdef int maxFreq = floor(log(maxUserRates, base))
    cdef np.ndarray[double, ndim=2] freqOffset = np.zeros([nMov, maxFreq + 1], 'float')
    cdef np.ndarray[uint, ndim=2] freqOffsetCount = np.zeros([nMov, maxFreq + 1], 'uint32')
    for i in range(nTrain):
        user = userdata[i] - 1
        movie = moviedata[i] - 1
        freq = floor(log(userRates[user], base))
        freqOffset[movie, freq] += ratingdata[i] - pred_archive[i]
        freqOffsetCount[movie, freq] += 1
    for i in range(nMov):
        for j in range(maxFreq + 1):
            freqOffset[i, j] = freqOffset[i, j] / (freqOffsetCount[i, j] + w * 0.1)
    np.savetxt(ctn.freqOffsetPath(), freqOffset)
    print "Max frequency: ", maxFreq
    print "Computing frequency offset took: ", time.time() - start_time, "seconds"
            
    start_time = time.time()
    # Compute the mean square error
    cdef double totalErr = 0
    cdef double diff
    for i in range(nTrain):
        user = userdata[i] - 1
        movie = moviedata[i] - 1
        pred = movieAvg[movie] + userOffset[user]
        for j in range(nFeat):
            pred += movieFeature[movie, j] * userFeature[user, j]
        diff = pred - ratingdata[i]
        totalErr += (diff * diff)
    cdef double meanErr = 0
    meanErr = sqrt(totalErr / nTrain)
    print "EIn: ", meanErr
    print "Computing EIn took: ", time.time() - start_time, "seconds"
    userFeature.tofile(ctn.SVDUserFeatPath())
    movieFeature.tofile(ctn.SVDMovieFeatPath())
    
