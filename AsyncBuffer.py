import numpy as np

class AsyncBuffer:

    def __init__(self, capacity, dtype):
        self.capacity   = capacity
        self.dtype      = dtype
        self.buffer     = np.empty(capacity, dtype=dtype)
        self.headIndex  = 0
        self.tailIndex  = 0

    def _read(self, nElements: int, peek):
        retIndex  = 0
        headIndex = self.headIndex
        ret       = np.empty(nElements, dtype = self.dtype)

        while nElements:
            cBufferLeft = self.capacity - headIndex
            elementsToRead = min(nElements, cBufferLeft)
            ret[retIndex: retIndex + elementsToRead] = self.buffer[headIndex : headIndex + elementsToRead]
            nElements   -= elementsToRead
            retIndex    += elementsToRead
            headIndex   += elementsToRead
            if headIndex == self.capacity:
                headIndex = 0

        if not peek:
            self.headIndex = headIndex
        return ret    

    def read(self, nElements: int, overlap: int = 0):
        ret = np.empty(nElements, dtype = self.dtype)
        ret[:overlap] = self._read(overlap, True)
        nonOverlappedElements = nElements - overlap
        ret[overlap : overlap + nonOverlappedElements] = self._read(nonOverlappedElements, False)
        return ret

    def write(self, a):
        cElementsLeft = len(a)
        aIndex = 0
        while cElementsLeft:
            cBufferLeft = self.capacity - self.tailIndex
            elementsToWrite = min(cElementsLeft, cBufferLeft)
            self.buffer[self.tailIndex : self.tailIndex + elementsToWrite] = a[aIndex : aIndex + elementsToWrite]
            cElementsLeft -= elementsToWrite
            aIndex += elementsToWrite
            self.tailIndex += elementsToWrite
            if self.tailIndex == self.capacity:
                self.tailIndex = 0

    def reset(self):
        self.headIndex = 0
        self.tailIndex = 0

    def numUnreadSamples(self):
        tailIndex = self.tailIndex
        if tailIndex < self.headIndex:
            tailIndex = self.capacity + tailIndex
        return tailIndex - self.headIndex
