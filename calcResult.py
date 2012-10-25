import IOmodule as iom
import numpy as np
import time

def calcResult():
    numF = 2
    user_f = np.fromfile('uf', 'float')
    movie_f = np.fromfile('mf', 'float')
    toRate = np.fromfile('../binary/qual.dta', 'uint')
    rating = np.empty((2749898), 'float')
    movie_a = np.fromfile('../binary/movie_a.dta', 'float')
    user_a = np.fromfile('../binary/user_a.dta', 'float')
    start_time = time.time()
    for i in range(0, 8249694, 3):
        movie = toRate[i+1] - 1
        user = toRate[i] - 1
        #mf = []
        #for j in range(numF):
        #    mf.append(movie_f[movie + j * 17770])
        uf = user_f[(user * numF):(user * numF + numF)]
        mf = movie_f[(movie * numF):(movie * numF + numF)]
        rating[i/3] = sum(uf[:] * mf[:]) + movie_a[movie]
    print time.time() - start_time, "seconds"    
    np.savetxt('mv.dta', rating)
        