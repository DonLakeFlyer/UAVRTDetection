import numpy as np

def vectorTranspose(a: np.ndarray):
    if a.ndim != 1:
        raise
    return a[:, np.newaxis]