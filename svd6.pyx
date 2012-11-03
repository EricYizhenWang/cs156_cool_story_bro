import IOmodule as iom
cimport numpy as np
import numpy as np
import time
cimport cython
from math import sqrt 

@cython.boundscheck(False)
@cython.wraparound(False)

def svd():
    cdef int i, j, k, p, user, movie, numIter, numF, numU, numM, numE
    cdef double lrate, pred, delta, reg, u, m, temp
    lrate = 0.001
    reg = lrate * 0.02
    numIter = 125
    numF = 50
    numU = 458293
    numM = 17770
    numE = 94362233
    cdef np.ndarray[double, ndim=2] userFeature = np.zeros([numU, numF], 'float') + 0.1
    cdef np.ndarray[double, ndim=2] movieFeature = np.zeros([numM, numF], 'float') + 0.1
    cdef np.ndarray[uint, ndim=1] userTemp = np.empty(numE, 'uint32')
    cdef np.ndarray[ulong, ndim=1] userTemp2 = np.empty(numE, 'uint')
    cdef np.ndarray[uint, ndim=1] movieTemp = np.empty(numE, 'uint32')
    cdef np.ndarray[ulong, ndim=1] movieTemp2 = np.empty(numE, 'uint')
    cdef np.ndarray[char, ndim=1] ratingTemp = np.empty(numE, 'int8')
    cdef np.ndarray[ulong, ndim=1] ratingTemp2 = np.empty(numE, 'uint')
    cdef np.ndarray[double, ndim = 1] movie_a = np.fromfile('../binary/movie_a.dta', 'float')
    cdef np.ndarray[double, ndim = 1] user_a = np.fromfile('../binary/user_a.dta', 'float')
    cdef np.ndarray[float, ndim = 1] pred_archive = np.empty(numE, 'float32')
    
    userTemp2 = np.fromfile('../binary/user.dta', 'uint')
    for i in range(numE):
        userTemp[i] = userTemp2[i]
    userTemp2 = np.empty(numE, 'uint')
    
    movieTemp2 = np.fromfile('../binary/movie.dta', 'uint')
    for i in range(numE):
        movieTemp[i] = movieTemp2[i]
    movieTemp2 = np.empty(numE, 'uint')
        
    ratingTemp2 = np.fromfile('../binary/date.dta', 'uint')
    for i in range(numE):
        ratingTemp[i] = ratingTemp2[i]
    ratingTemp2 = np.empty(numE, 'uint')
    
    start_time = time.time()
    # Calculate a good baseline
    # BetterMean = [GlobalAverage*K + sum(ObservedRatings)] / [K + count(ObservedRatings)]
    cdef double globalTotal = 0
    cdef np.ndarray[ushort, ndim=1] userRates = np.empty(numU, 'uint16')
    cdef np.ndarray[ushort, ndim=1] movieRates = np.empty(numM, 'uint16')
    cdef np.ndarray[float, ndim=1] movieAvg = np.empty(numM, 'float32')
    cdef np.ndarray[float, ndim=1] userAvg = np.empty(numU, 'float32')
    cdef np.ndarray[float, ndim=1] offset = np.empty(numU, 'float32')
    
    cdef double globalAvg = 0
        
    cdef int w = 25    
    for i in range(numE):
        user = userTemp[i]-1
        movie = movieTemp[i]-1
        globalTotal += ratingTemp[i]
        userRates[user] += 1
        movieRates[movie] += 1
        
    globalAvg = globalTotal / numE
    
    cdef double totalRating = 0
    # Compute the smoothed movie_average
    for i in range(numM):
        totalRating = movie_a[i] * movieRates[i]
        movieAvg[i] = (globalAvg * w + totalRating) / (w + movieRates[i])
    
    for i in range(numU):
        totalRating = user_a[i] * userRates[i]
        userAvg[i] = (globalAvg * w + totalRating) / (w + userRates[i])
        
    # Compute the user offset
    for i in range(numE):
        user = userTemp[i]-1
        movie = movieTemp[i]-1
        offset[user] += ratingTemp[i] - movie_a[movie]
    for i in range(numU):
        offset[i] = offset[i] / (userRates[i] + (w * 1.0))

    print time.time() - start_time, "seconds"
    
    np.savetxt('../binary/movieAvg_dta', movieAvg)
    np.savetxt('../binary/userAvg_dta', userAvg)
    np.savetxt('../binary/offset', offset)
    print globalAvg
    
    
    start_time = time.time()
    cdef np.ndarray[double, ndim=1] movief = np.empty(numM, 'float')
    cdef np.ndarray[double, ndim=1] userf = np.empty(numU, 'float')
    
    for i in range(numE):
        user = userTemp[i]-1
        movie = movieTemp[i]-1
        pred_archive[i] = movieAvg[movie] + offset[user]
        
    for j in range(numF):
        for i in range(numU):
            userf[i] = userFeature[i, j]
        for i in range(numM):   
            movief[i] = movieFeature[i, j]    
        for k in range(numIter):     
            for i in range(numE):   
                user = userTemp[i]-1
                movie = movieTemp[i]-1
                u = userf[user]
                m = movief[movie]           
                temp = pred_archive[i] + u * m
                delta = ratingTemp[i] - temp
                err = lrate * delta
                userf[user] = u + (err * m - reg * u) 
                movief[movie] = m + (err * u - reg * m)
        for i in range(numU):
            userFeature[i, j] = userf[i]
        for i in range(numM):
            movieFeature[i, j] = movief[i]
        for i in range(numE):
            user = userTemp[i]-1
            movie = movieTemp[i]-1
            pred_archive[i] += userf[user] * movief[movie]
            
    print time.time() - start_time, "seconds"
    start_time = time.time()
    # Compute the mean square error
    cdef double totalErr = 0
    cdef double diff
    for i in range(numE):
        user = userTemp[i]-1
        movie = movieTemp[i]-1
        pred = movieAvg[movie] + offset[user]
        for j in range(numF):
            pred += movieFeature[movie, j] * userFeature[user, j]
        diff = pred - ratingTemp[i]
        totalErr += (diff * diff)
    cdef double meanErr = 0
    meanErr = sqrt(totalErr / numE)
    print meanErr
    print time.time() - start_time, "seconds"
    userFeature.tofile('uf')
    movieFeature.tofile('mf')
    