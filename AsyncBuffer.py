import numpy as np

class AsyncBuffer:

    def __init__(self, capacity, dataType):
        self.capacity           = capacity
        self.dataType           = dataType
        self.buffer             = np.empty(capacity, dtype = dataType)
        self.numUnreadSamples   = 0

    def write(self, array):
        if not isinstance(array, np.ndarray):
            raise Exception("array must be of type NumPy.ndarray")
        np.append(self.buffer, array)