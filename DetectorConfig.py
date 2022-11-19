import configparser

class DetectorConfig:
#    %DETECTORCONFIG Class contains the information needed by the detector
#    %to process a channel of radio data. 
#    %
#    %PROPERTIES
#    %   ID              A string identifier for the detector
#    %   channelCenterFreqMHZ    Center frequency of incoming data stream in MHz
#    %   ipData          String ip from which to receive data. Enter
#    %                   '0.0.0.0' to receive from any IP.
#    %   portData        Port from which to receive data
#    %   ipCntrl         String ip from which to receive control inputs. 
#    %                   Enter '0.0.0.0' to receive from any IP.
#    %   portCntrl       Port from which to receive control inputs
#    %   Fs              Sample rate of incoming data
#    %   tagFreqMHz      Expected frequency of tag
#    %   tp              Duration of pulse in seconds
#    %   tip             Interpulse duration in seconds
#    %   tipu            Interpulse duration uncertainty in seconds
#    %   tipj            Interpulse duration jitter in seconds
#    %   K               Number of pulses to integrate
#    %   opMode          Operational mode for processing:
#    %                       freqSearchHardLock
#    %                       freqSearchSoftLock
#    %                       freqKnownHardLock
#    %                       freqAllNeverLock

    def __init__ (self):
        tagSection = "Tag";
        config = configparser.ConfigParser()
        config.read("DetectorConfig.ini")

        self.tagId                  = config[tagSection].getint("id")
        self.channelCenterFreqMHz   = config[tagSection].getfloat("centerFreqMHz")
        self.ipData                 = config[tagSection]["dataIp"]
        self.portData               = config[tagSection].getint("dataPort")
        self.Fs                     = config[tagSection].getfloat("sampleRateHz")
        self.tagFreqMHz             = config[tagSection].getfloat("tagFreqMHz")
        self.tp                     = config[tagSection].getfloat("pulseDurationSecs")
        self.tip                    = config[tagSection].getfloat("intraPulseSecs")
        self.tipu                   = config[tagSection].getfloat("intraPulseUncertaintySecs")
        self.tipj                   = config[tagSection].getfloat("intraPulseJitterSecs")
        self.K                      = config[tagSection].getint("k")
        self.opMode                 = config[tagSection]["opMode"]
        self.falseAlarmProb         = config[tagSection].getfloat("falseAlarmProb")

    def __str__(self): 
        return "DetectorConfig(%x): tagId:%d channelCenterFreqMHz:%f ipData:%s, portData:%d Fs:%f tp:%f tip:%f tipu:%f tipj:%f K:%d" % \
            (id(self), self.tagId, self.channelCenterFreqMHz, self.ipData, self.portData, self.Fs, self.tp, self.tip, self.tipu, self.tipj, self.K)