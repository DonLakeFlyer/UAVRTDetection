import numpy as np
import scipy
import math

def matlabRange(start: int, stopInclusive: int):
    # Meant to be used to convert matlab syntax in the form (x: n)
    return np.arange(start, stopInclusive + 1)

def norm(v: np.ndarray):
    if v.ndim != 1:
        raise Exception("Only vectors are supported")
    return np.linalg.norm(v, 2)

def reshape(x: np.ndarray, sz1: int, sz2: int):
    return  x.reshape(sz1, sz2, order='F').copy()

def find(a: np.ndarray, n: int, direction = 'first'):
    if a.ndim != 1:
        raise Exception("Only 1d supported")
    first = 'first'
    last = 'last'
    if not direction in (first, last):
        raise
    indices  = np.nonzero(a)
    if n == 1:
        if direction == first:
            return indices[0][0]
        else:
            return indices[0][-1]
    else:
        # NYI
        raise

def sort(v: np.ndarray):
    if v.ndim != 1:
        raise Exception("Only vectors are supported")
    sortIndices = np.argsort(v)
    sortedArray = np.take_along_axis(v, sortIndices, 0)
    return [ sortedArray, sortIndices ]

def numel(a: np.ndarray):
    return np.size(a)

def mod(a, m):
    return math.fmod(a, m)

# Return a "stop inclusive" vector
#   matlab - 1:4            -> 1, 2, 3, 4
#   numpy  - np.arange(1,4) -> 1, 2, 3
#            siVector(1,4)  -> 1, 2, 3, 4
def siVector(start: int, stopInclusive: int):
    # Meant to be used to convert matlab syntax in the form (x: n)
    return np.arange(start, stopInclusive + 1)
