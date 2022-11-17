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
    
    def __init__(self, amplitude, yw, snr, startTime, endTime, nextTimeRange, peakFreq, freqStart, freqEnd):
        self.amplitude      = amplitude         # A
        self.yw             = yw                # yw
        self.snr            = snr               # SNR
        self.startTime      = startTime         # t_0
        self.endTime        = endTime           # t_f
        self.nextTimeRange  = nextTimeRange     # t_next
        self.peakFreq       = peakFreq          # fp
        self.freqStart      = freqStart         # fstart
        self.freqEnd        = freqEnd           # fend
