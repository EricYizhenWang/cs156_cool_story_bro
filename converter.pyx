cimport cython
cimport numpy as np
import numpy as np

def convertIntToInt32(oldPath, newPath, length):
    cdef np.ndarray[ulong, ndim=1] oldFile = np.empty(length, 'uint')
    cdef np.ndarray[uint, ndim=1] newFile = np.empty(length, 'uint32')
    
    oldFile = np.fromfile(oldPath, 'uint')
    for i in range(length):
        newFile[i] = oldFile[i]
    newFile.tofile(newPath)

def convertIntToInt8(oldPath, newPath, length):
    cdef np.ndarray[ulong, ndim=1] oldFile = np.empty(length, 'uint')
    cdef np.ndarray[char, ndim=1] newFile = np.empty(length, 'int8')
    
    oldFile = np.fromfile(oldPath, 'uint')
    for i in range(length):
        newFile[i] = oldFile[i]
    newFile.tofile(newPath)

def convertInt32ToInt8(oldPath, newPath, length):
    cdef np.ndarray[uint, ndim=1] oldFile = np.empty(length, 'uint32')
    cdef np.ndarray[char, ndim=1] newFile = np.empty(length, 'int8')

    oldFile = np.fromfile(oldPath, 'uint32')
    for i in range(length):
        newFile[i] = oldFile[i]
    newFile.tofile(newPath)
