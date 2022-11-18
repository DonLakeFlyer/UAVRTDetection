import numpy as np
import bottleneck

def movingMean(a: np.ndarry, k: int, axis: int = 0):
    # Example:
    #   movingMean(a, k) equates to matlab movmean(a, k)
    #   movingMean(a, k, axis = 0) equates to matlab movmean(a, k, 1)
    return bottleneck.move_mean(a, window = k, axis = axis)
