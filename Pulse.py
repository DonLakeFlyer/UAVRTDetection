import numpy as np

class Pulse:
#    properties
#        A           %Amplitude
#        P           %Power
#        SNR         %Signal to noise ratio in dB - This is often estimated as signal+noise to noise. 
#        yw          %STFT value at location (time and freq) of pulse. w here mean omega. This is equation 14 in the paper draft.
#        t_0         %Start time of pulse
#        t_f         %End time of pulse
#        t_next      %Time range [tstart tend] of expected location of next pulse
#        fp          %Peak frequency of pulse (Center frequency if symmetric in frequency domain
#        fstart      %Start of the frequency bandwidth
#        fend        %End of frequency bandwidth
#        mode        %State machine mode under which pulse was discovered
#        det_dec     %Detection decision (true/false)
#        con_dec     %Was the pulse confirmed (true/false). In tracking, no confirmation step is executed so we record false.
#    end
    
    def __init__(self, 
                    amplitude:      float       = np.nan, 
                    yw:             float       = np.nan, 
                    snr:            float       = np.nan, 
                    startTime:      float       = np.nan, 
                    endTime:        float       = np.nan, 
                    nextTimeRange:  np.ndarray  = np.array((np.nan, np.nan)), 
                    peakFreq:       float       = np.nan, 
                    freqStart:      float       = np.nan, 
                    freqEnd:        float       = np.nan):
        self.A          = amplitude         # A
        self.P          = self.A ** 2
        self.yw         = yw                # yw
        self.SNR        = snr               # SNR
        self.t_0        = startTime         # t_0
        self.t_f        = endTime           # t_f
        self.t_next     = nextTimeRange     # t_next
        self.fp         = peakFreq          # fp
        self.fstart     = freqStart         # fstart
        self.fend       = freqEnd           # fend
        self.mode       = "TBD"
        self.det_dec    = False
        self.con_dec    = False

    def pulseArrayAnyConfirmed(pulseArray):
        for pulse in pulseArray.flat:
            if pulse.con_dec:
                return True
        return False

    def pulseArrayAnyDetected(pulseArray):
        for pulse in pulseArray.flat:
            if pulse.det_dec:
                return True
        return False
