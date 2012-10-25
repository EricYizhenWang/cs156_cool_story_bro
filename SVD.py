import numpy as np
from IOModule import *
from constants import *

def svd(rate, n, dim):
    userdata = read(userPath(), np.dtype('i4'))
    moviedata = read(moviePath(), np.dtype('i4'))
    ratingdata = read(ratingPath(), np.dtype('i4'))
    userfeat = np.empty((userNumber(), dim))
    userfeat.fill(0.1)
    moviefeat = np.empty((movieNumber(), dim))
    moviefeat.fill(0.1)
    for i in range(n):
        for j in range(trainSize()):
            userj = userdata[j]
            moviej = moviedata[j]
            ratingj = ratingdata[j]
            prediction = np.dot(userfeat[userj], moviefeat[moviej])
            error = ratingj - prediction
            userfeat[userj] += error * rate * moviefeat[moviej]
            moviefeat[moviej] += error * rate * userfeat[userj]
    printToBin(userfeat, SVDUserFeatPath())
    printToBin(moviefeat, SVDMovieFeatPath())

if __name__ == '__main__':
    svd(0.001, 100, 40)