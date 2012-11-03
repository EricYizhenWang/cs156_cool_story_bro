import IOmodule as iom
cimport numpy as np
import numpy as np
import time
cimport cython

def calcResult():
    cdef int i, j, k, movie, user, numF, numU, numE, numM
    cdef double globalAvg, a, temp
    numF = 50
    numU = 458293
    numM = 17770
    numE = 94362233
    cdef np.ndarray[double, ndim = 1] user_f = np.fromfile('uf', 'float')
    cdef np.ndarray[double, ndim = 1] movie_f = np.fromfile('mf', 'float')
    cdef np.ndarray[ulong, ndim = 1] toRate = np.fromfile('../binary/qual.dta', 'uint')
    cdef np.ndarray[double, ndim = 1] rating = np.empty((2749898), 'float')
    cdef np.ndarray[double, ndim = 1] movieAvg = np.empty((numM), 'float')
    cdef np.ndarray[double, ndim = 1] userAvg = np.empty((numU), 'float')
    cdef np.ndarray[double, ndim = 1] offset = np.empty((numU), 'float')
    cdef np.ndarray[double, ndim = 1] movie_a = np.fromfile('../binary/movie_a.dta', 'float')
    cdef np.ndarray[double, ndim = 1] user_a = np.fromfile('../binary/user_a.dta', 'float')
    movieAvg = np.loadtxt('../binary/movieAvg_dta')
    userAvg = np.loadtxt('../binary/userAvg_dta')
    offset = np.loadtxt('../binary/offset')
    globalAvg = 3.60860890183
    print movie_f[:10], user_f[:10] 
    
    start_time = time.time()
    for i in range(0, 8249694, 3):
        movie = toRate[i+1] - 1
        user = toRate[i] - 1
        #a = movieAvg[movie] + userAvg[user] - globalAvg
        a = movieAvg[movie] + offset[user]
        #a = movie_a[movie] + user_a[user] - globalAvg
        #a = 0.8 * movie_a[movie] + 0.2 * user_a[user]
        for j in range(numF):
            a += (movie_f[movie*numF + j] * user_f[user*numF + j])
            #a += temp[j]
            if (a > 5):
                a = 5
            elif (a < 1):
                a = 1
        rating[i/3] = a
    print time.time() - start_time, "seconds"
    
    np.savetxt('mv.dta', rating)
        