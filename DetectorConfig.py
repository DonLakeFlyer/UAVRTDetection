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

        self.tagId                      = config[tagSection].getint("tagId")
        self.centerFreq                 = config[tagSection].getfloat("centerFreqMHz")
        self.dataIp                     = config[tagSection]["dataIp"]
        self.dataPort                   = config[tagSection].getint("dataPort")
        self.sampleRateHz               = config[tagSection].getfloat("sampleRateHz")
        self.tagFreqMHz                 = config[tagSection].getfloat("tagFreqMHz")
        self.pulseDurationSecs          = config[tagSection].getfloat("pulseDurationSecs")
        self.intraPulseSecs             = config[tagSection].getfloat("intraPulseSecs")
        self.intraPulseUncertaintySecs  = config[tagSection].getfloat("intraPulseUncertainty")
        self.intraPulseJitterSecs       = config[tagSection].getfloat("intraPulseJitter")
        self.k                          = config[tagSection].getint("k")
        self.opMode                     = config[tagSection]["opMode"]
        self.falseAlarmProb             = config[tagSection].getfloat("falseAlarmProb")
