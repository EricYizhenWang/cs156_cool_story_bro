import numpy as np
import constants as cons

#
# Bogus module for finding out statistics about the data
#

def userRange():
    userdata = np.fromfile(cons.userPath(), np.dtype('uint32'))
    return (np.min(userdata), np.max(userdata))

def movieRange():
    moviedata = np.fromfile(cons.moviePath(), np.dtype('uint32'))
    return (np.min(moviedata), np.max(moviedata))

if __name__ == '__main__':
    print('userRange(): ', userRange())
    print('movieRange(): ', movieRange())
