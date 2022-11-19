import numpy as np

class AsyncBuffer:

    def __init__(self, capacity, dataType):
        self.capacity           = capacity
        self.dataType           = dataType
        self.buffer             = np.empty(capacity, dtype = dataType)

    def read(self, nElements: int):
        return self.buffer[:nElements]

    def write(self, array):
        if not isinstance(array, np.ndarray):
            raise
        np.append(self.buffer, array)

    def reset(self):
        # FIXME: NYI
        pass

    def numUnreadSamples(self):
        return len(self.buffer)