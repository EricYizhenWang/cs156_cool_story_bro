#import IOmodule as iom
cimport numpy as np
import numpy as np
import time
cimport cython
#from scipy.sparse import coo_matrix

@cython.boundscheck(False)
@cython.wraparound(False)

def inputs():
    numF = 10
    numU = 458293
    numM = 17770
    numE = 94362233
    userTemp = np.fromfile('../binary/user.dta', 'uint')
    movieTemp = np.fromfile('../binary/movie.dta', 'uint')
    ratingTemp = np.fromfile('../binary/date.dta', 'uint')
    cdef np.ndarray[double, ndim=2] userFeature = np.zeros([numU, numF], 'float') + 0.1
    cdef np.ndarray[double, ndim=2] movieFeature = np.zeros([numM, numF], 'float') + 0.1
    return userTemp, movieTemp, ratingTemp, userFeature, movieFeature
    
def svd(np.ndarray[double, ndim=2] userFeature,
        np.ndarray[double, ndim=2] movieFeature,
        np.ndarray[ulong, ndim=1] userTemp,
        np.ndarray[ulong, ndim=1] movieTemp,
        np.ndarray[ulong, ndim=1] ratingTemp):
    cdef int i, j, k, p, user, movie, numIter, numF, numU, numM, numE
    cdef double lrate, pred, delta
    lrate = 0.002
    numIter = 1
    numF = 10
    numU = 458293
    numM = 17770
    numE = 94362233
    
    #movie_a = np.fromfile('../binary/movie_a.dta', 'float')
    start_time = time.time()
    
    for k in range(numIter):
        for j in range(numF):        
            for i in range(numE):
                user = userTemp[i]-1
                movie = movieTemp[i]-1
                pred = 0
                for p in range(j+1):
                    pred += userFeature[user, p] * movieFeature[movie, p]
                #temp = userFeature[user, :(j+1)] * movieFeature[movie, :(j+1)]
                #pred = sum(temp) #+ movie_a[movie]
                delta = lrate * (ratingTemp[i] - pred)
                print delta
                userFeature[user, j] += delta * movieFeature[movie, j] 
                movieFeature[movie, j] += delta * userFeature[user, j]
            
    print time.time() - start_time, "seconds"
    userFeature.tofile('uf')
    movieFeature.tofile('mf')
        
def running():
    (userTemp, movieTemp, ratingTemp, userFeature, movieFeature) = inputs()
    svd(userFeature, movieFeature, userTemp, movieTemp, ratingTemp)