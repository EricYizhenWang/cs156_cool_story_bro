cimport numpy as np
import numpy as np
cimport cython

def convertIntToInt32Dim1(file, newPath, length):
    cdef np.ndarray[uint, ndim=1] oldFile = np.empty(length, 'uint')
    cdef np.ndarray[uint, ndim=1] newFile = np.empty(length, 'uint32')
    
    oldFile = np.fromfile(file, 'uint')
    for i in range(length):
        newFile[i] = oldFile[i]
    newFile.tofile(newPath)
    
def convertIntToInt32Dim2(file, newPath, dim):
    cdef np.ndarray[uint, ndim=2] oldFile = np.empty(dim, 'uint')
    cdef np.ndarray[uint, ndim=2] newFile = np.empty(dim, 'uint32')
    
    oldFile = np.fromfile(file, 'uint')
    for i in range(dim[0]):
        for j in range(dim[1]):
            newFile[i, j] = oldFile[i, j]
    newFile.tofile(newPath)