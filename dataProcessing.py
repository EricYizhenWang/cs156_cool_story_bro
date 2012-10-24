import numpy as np
from IOModule import *
from constants import *

#
# Bogus module for finding out statistics about the data
#

def userRange():
    userdata = read(userPath(), np.dtype('i4'))
    return (np.min(userdata), np.max(userdata))

def movieRange():
    moviedata = read(moviePath(), np.dtype('i4'))
    return (np.min(moviedata), np.max(moviedata))

if __name__ == '__main__':
    print('userRange(): ', userRange())
    print('movieRange(): ', movieRange())