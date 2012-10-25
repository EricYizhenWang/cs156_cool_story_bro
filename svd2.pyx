cimport numpy as np
import numpy as np
import time
cimport cython
from scipy.sparse import coo_matrix

@cython.boundscheck(False)
@cython.wraparound(False)

def svd():
    # static typing
    cdef int i, j, k, p, user, movie, numIter, numF, numU, numM, numE
    cdef double lrate, pred, delta
    # define constants
    lrate = 0.001   
    numIter = 10    
    numF = 2        
    numU = 458293
    numM = 17770
    numE = 94362233 # no. of entries in the input data
    # more static typing
    cdef np.ndarray[double, ndim=2] userFeature = np.zeros([numU, numF], 'float') + 0.1
    cdef np.ndarray[double, ndim=2] movieFeature = np.zeros([numM, numF], 'float') + 0.1
    cdef np.ndarray[ulong, ndim=1] userTemp = np.empty(numE, 'uint')
    cdef np.ndarray[ulong, ndim=1] movieTemp = np.empty(numE, 'uint')
    cdef np.ndarray[ulong, ndim=1] ratingTemp = np.empty(numE, 'uint')
    # load data
    userTemp = np.fromfile('../binary/user.dta', 'uint')
    movieTemp = np.fromfile('../binary/movie.dta', 'uint')
    ratingTemp = np.fromfile('../binary/date.dta', 'uint')
    movie_a = np.fromfile('../binary/movie_a.dta', 'float')
    
    start_time = time.time()    
    for k in range(numIter):    
        for i in range(numE):
            user = userTemp[i]-1
            movie = movieTemp[i]-1
            pred = movie_a[movie]
            for p in range(numF):
                pred += userFeature[user, p] * movieFeature[movie, p]
            delta = lrate * (ratingTemp[i] - pred)
            for p in range(numF):
                userFeature[user, p] += delta * movieFeature[movie, p] 
                movieFeature[movie, p] += delta * userFeature[user, p]
        
    print time.time() - start_time, "seconds"
    userFeature.tofile('uf')
    movieFeature.tofile('mf')
    