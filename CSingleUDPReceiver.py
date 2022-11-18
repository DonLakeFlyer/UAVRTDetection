import socket
import numpy
import sys

class CSingleUDPReceiver:
    def __init__ (self, ipAddress: str, port: int):
        self.rxSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rxSocket.bind((ipAddress, port))

    def read(self, complexSampleCount: int):
        # complexSampleCount: buffer size in complex samples
        bytes = self.rxSocket.recvfrom(complexSampleCount * sys.sizeof(numpy.csingle()))
        return numpy.frombuffer(bytes, dtype=  numpy.csingle)

    def clear(self):
        self.rxSocket.setblocking(False)
        while len(self.rxSocket.recvfrom(1024)[0]) != 0:
            pass
        self.rxSocket.setblocking(True)