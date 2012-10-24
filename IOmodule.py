import numpy as np

# This method takes the filename of a binary file and also the type
# of the content in the file. It returns a numpy array which contains
# the file content
def read(filename, typename):
    a = np.fromfile(filename, typename)
    return a

# This method takes an nparray and output it to a text file.
def printToTxt(nparray, filename):
    np.savetxt(filename, nparray)
    
    