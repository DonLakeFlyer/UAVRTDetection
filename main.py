import DetectorConfig
import socket
import NumPy

def main():
    # Load configuration
    config = DetectorConfig.DetectorConfig()
    print("%s %d" % (config.dataIp, config.dataPort))

    rxSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rxSocket.bind((config.dataIp, config.dataPort))

    timeStamp = 0
    while True:
        data = rxSocket.recv(1024)
        print("Received data %d" % len(data))

        #timeVector     = timeStamp+1/Config.Fs*(0:(numel(iqData)-1)).';
        #timeStamp      = timeStamp + (numel(iqData) / Config.Fs);


if __name__ == '__main__':
    main()