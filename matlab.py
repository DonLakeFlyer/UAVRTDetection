import numpy as np
import scipy
import math

def circshift(a: np.ndarray, K: int, dim: int):
    # Note: dim follows matlab 1-based indexing
    if a.ndim > 2:
        raise Exception("Only 1D or 2D is supported")
    return np.roll(a, -K, axis = dim - 1)

def fftshift(a: np.ndarray, dim: int):
    # Note: dim follows matlab 1-based indexing
    if a.ndim > 2:
        raise Exception("Only 1D or 2D is supported")
    return scipy.fft.fftshift(a, axis = dim - 1)

def ifftshift(a: np.ndarray, dim: int):
    # Note: dim follows matlab 1-based indexing
    if a.ndim > 2:
        raise Exception("Only 1D or 2D is supported")
    return scipy.fft.ifftshift(a, axis = dim - 1)

def matlabRange(start: int, stopInclusive: int):
    # Meant to be used to convert matlab syntax in the form (x: n)
    np.arange(start, stopInclusive + 1)

def norm(v: np.ndarray):
    if v.ndim != 1:
        raise Exception("Only vectors are supported")
    return np.linalg.norm(v, 2)

def reshape(x: np.ndarray, sz1: int, sz2: int):
    return  x.reshape(sz1, sz2, order='F').copy()

def find(a: np.ndarray, n: int, direction = 'first'):
    first = 'first'
    last = 'last'
    if not direction in (first, last):
        raise
    indices  = np.flatnonzero(a)
    if n == 1:
        if direction == first:
            return indices[0][0]
        else:
            return indices[0][1]
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