import numpy as np

def matlabFind(a: np.ndarray, n: int, direction = 'first'):
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
