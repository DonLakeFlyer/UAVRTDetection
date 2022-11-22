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

def find(a: np.ndarray, n: int = None, direction = 'first'):
    if a.ndim != 1:
        raise Exception("Only 1d supported")
    first = 'first'
    last = 'last'
    if not direction in (first, last):
        raise
    if n is None:
        return a.ravel().nonzero()
    elif n == 1:
        indices  = np.nonzero(a)
        if direction == first:
            index = 0
        else:
            index = -1
        if len(indices[0]) == 0:
            return None
        else:
            return indices[0][index]
    else:
        # NYI
        raise

# Returns a vector
def findV(a: np.ndarray, n: int = None, direction = 'first'):
    ret = find(a, n, direction)
    if ret is None:
        return np.array([])
    else:
        return np.array([ret])

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
    return np.arange(start, stopInclusive + 1, dtype=np.int_)

def interp1(x, v, xq, kind = 'linear', fill_value = None):
    # scipy.interpolate.interp1d does not support complex data
    # Split into parts and put back together
    if fill_value == 'extrap':
        fill_value = 'extrapolate'
    xqReal = np.real(xq)
    xqImag = np.imag(xq)
    func = scipy.interpolate.interp1d(x, v, kind = kind, fill_value = fill_value)
    return func(xqReal) + func(xqImag)*1j

def numel(a):
    if type(a) is list:
        return len(a)
    else:
        return a.size

def repmat(A, r1, r2):
    return np.tile(A, (r1, r2))

def size(a, dim = None):
    if not isinstance(a, np.ndarray):
        raise
    if not isinstance(dim, int):
        raise
    if dim is None:
        return a.shape
    else:
        if dim > 2:
                raise
        return a.shape[dim-1]

def sum(a, dim: None):
    if dim is None:
        if a.shape[0] == 1:
            raise
        dim = 0
    elif isinstance(dim, int):
        if dim > 2:
            raise
        axis = dim - 1
    elif dim == 'all':
        axis = None
    else:
        raise
    return a.sum(axis = axis)

def transposeV(v):
    return v.reshape(-1,1)

def sub2indDeprecated(shp, row, col):
    n_rows = shp[0]
    return [(n_rows * (c-1)) + r - 1 for r, c in zip(row, col)]

def circshift(a, k, dim = None):
    if not isinstance(k, int):
        raise
    if dim is None:
        dim = find(a.shape != 1, 1, 'first') + 1
    return np.roll(a, k, dim - 1)

def mean(a, dim):
    return np.mean(a, axis = dim -1)

def makeColVector(r):
    if r.ndim != 1:
        raise Exception("Input must be row vector")
    return r.reshape(-1,1)