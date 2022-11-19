from movingMean import movingMean
from vectorTranspose import vectorTranspose

import scipy
import numpy as np

# classdef wfmstft < handle
class WaveformSTFT:
    # WFMSTFT is a class designed to containt a short-time Fourier 
    # transform and its associated descriptive properties
    # 
    # 
    # This class will build and hold and STFT for a waveform pddrovided to it
    # as an input. It will also hold the time and frequency vectors for the
    # resulting STFT matrix.
    #    
    # PROPERTIES:
    #    S           Spectrogram ouput
    #    f           Frequency vector for rows of S
    #    t           Time vector for columns of S
    #    wind_per_pulse  number of STFT time windows per pulse as described
    #                     in the waveform properties
    #    stft_steps_per_pulse    # number of stft time windpws
    #    dt          1/Fs of the data used to generate the STFT. This is
    #                needed for the PSD calculation.
    #    T           Time duration of an STFT window (n_w/Fs). This is
    #                needed for the PSD calculation.
    #    psd         A noise-only power spectral density estimate that based
    #                on a moving average of the S matrix, with thresholding
    #                the exclude high-power pulses. 
    #    window      The window used for the STFT
    #    
    # ----------------------------------------------------------------------
    # Author: Michael W. Shafer
    # Date: 2020-05-28
    # 	
    # ----------------------------------------------------------------------
    # Updates:
    #    2022-03-31  Added psd, dt and T as properties. 
    # 
    # # 
    
    # properties
    #     S   (:,:) double	# Spectrogram ouput
    #     f   (:,1) double   # Frequency vector for rows of S
    #     t   (:,1) double   # Time vector for columns of S
    #     psd (:,1) double # Power spectral density (W/Hz)
    #     dt  (1,1) double  # 1/Fs of data used to generate STFT
    #     T   (1,1) double   # Time of each window of the STFT (n_w/Fs)
    #     wind(:,1) double # Window definition used for the STFT
    # end
    
    # methods
    #     function obj = wfmstft(waveform_obj)
    def __init__ (self, waveform_obj):
        # WFMSTFT Constructs and returns an instance of this class
        # 
        # An waveform object must be passed to this construction method
        # so that the constructor has access to the data vector, desired
        # overlap fraction, and priori pulse data, which is used to
        # develop the window sizes. 
        # INPUTS:
        #    waveform_obj   A single waveform object with prior
        #                   dependent properties set. 
        # OUTPUTS:
        #    obj             A wfmstft object
        # # 
        # 
        # The following are variable sized properties. To tell coder
        # that they may vary setup as a local variable size variable
        # first, then set.
        # Instructions on https://www.mathworks.com/help/simulink/ug/how-working-with-matlab-classes-is-different-for-code-generation.html
        # localS   = double(complex(zeros(0,0)));
        # localt   = double(zeros(0, 1));
        # localf   = double(zeros(0, 1));
        # localpsd = double(zeros(0, 1));
        # localwind= double(zeros(0, 1));
        # coder.varsize('localS',   [200, 2000], [1 1]);
        # coder.varsize('localt',   [2000,   1], [1 0]);
        # coder.varsize('localf',   [200,    1], [1 0]);
        # coder.varsize('localpsd', [200,    1], [1 0]);
        # coder.varsize('localwind',[200,    1], [1 0]);# maxFs*maxpulsewidth

        # Now actually assign them
        # self.S   = localS;
        # self.t   = localt;
        # self.f   = localf;
        # self.psd = localpsd;
        # self.wind= localwind;
        # self.dt  = 0;
        # self.T   = 0;

        # if nargin>0
        # self.wind = rectwin(waveform_obj.n_w);
        self.wind = scipy.signal.windows.boxcar(waveform_obj.n_w)

        # [S, self.f, local_time] = stft(waveform_obj.x,waveform_obj.Fs,'Window',self.wind,'OverlapLength',waveform_obj.n_ol,'FFTLength',waveform_obj.n_w);
        # FIXME: Need correct stft setup from Michael
        self.f, local_time, S = scipy.signal.stft(
                                        waveform_obj.x, 
                                        waveform_obj.Fs,
                                        window      = self.wind,
                                        noverlap    = waveform_obj.n_ol,
                                        nperseg     = waveform_obj.n_w,
                                        nfft        = waveform_obj.n_w)

        # self.S = double(S);#  Incoming x data in waveform is single precision, but sparse matrix multipliation later is only supported for double precision.
        # self.t = double(local_time)+waveform_obj.t_0; # Convert to global time of waveform. Double cast is needed because if x data in stft is single precision then the output t will be single as well.
        # self.dt = 1/waveform_obj.Fs;
        # self.T  = waveform_obj.n_w/waveform_obj.Fs;
        # self.updatepsd();

        #  Incoming x data in waveform is single precision, but sparse matrix multipliation later is only supported for double precision.
        self.S = S.astype(np.double)
        # Convert to global time of waveform. Double cast is needed because if x data in stft is single precision then the output t will be single as well.
        self.t  = local_time.astype(np.double) + waveform_obj.t_0; 
        self.dt = 1 / waveform_obj.Fs;
        self.T  = waveform_obj.n_w / waveform_obj.Fs;
        self.updatePSD();
        # end
        
    # function [] = adddata(obj,Snew,tnew)
    def addData(self, Snew: np.ndarray, tnew: np.ndarray):
        # ADDDATA Tacks on additional time windows to the existing STFT
        # object and ammends the time vector accordingly. 
        # INPUTS:
        #    Snew    STFT matrix to be appended with same number of
        #            frequency bins (rows) as self.S
        #    tnew    Time vector of columns of Snew
        # OUTPUT:
        #    none
        # # 

        # self.S = [self.S, Snew];
        # self.t = [self.t;tnew];
        # self.updatesd();
        self.S.append(Snew)
        self.t.append(tnew)
        self.updatePSD();
        
    # function [] = setfreqs(obj,freqs)
    def setFreqs(self, freqs):
        # SETFREQS Sets the frequency vector for the stft matrix rows. 
        # INPUTS:
        #    freqs   A vector of frequencies corresponding to the rows
        #            of the stft matrix S
        # OUTPUT:
        #    none
        # # 
        self.f = freqs;
        
    # function wfmstftout = copy(obj)
    def copy(self):
        out = WaveformSTFT.WaveformSTFT();
        out.S       = self.S.deepcopy()
        out.f       = self.f.deepcopy()
        out.t       = self.t.deepcopy()
        out.dt      = self.dt.deepcopy()
        out.T       = self.T.deepcopy()
        out.psd     = self.psd.deepcopy()
        out.wind    = self.wind.deepcopy()
        return out

    # function [] = updatepsd(obj)
    def updatePSD(self):
        # This block calculates a three window moving mean of the power
        # spectral density of the waveform and then thresholds that
        # moving mean for values greater than 10x the median. This
        # thresholding step reduced the energy from very high power
        # pulses that might be in the signal from affecting the PSD
        # estimate. It is assumed here that low signal power pulses will
        # only marginally affect the psd estimate. 

        # coder.varsize('magSqrd','movMeanMagSqrd','medMovMeanMagSqrd','medMovMeanMagSqrdMat','magSqrdMask')
        # magSqrd             = abs(self.S).^2;
        # movMeanMagSqrd      = movmean(magSqrd,3,2);
        # medMovMeanMagSqrd   = median(movMeanMagSqrd,2); # transpose(median(transpose(movMeanMagSqrd)));# median(rand(80,32767),2);
        # medMovMeanMagSqrdMat = repmat(medMovMeanMagSqrd,1,size(magSqrd,2));
        # magSqrdMask = magSqrd > 10 * medMovMeanMagSqrdMat;
        # magSqrd(magSqrdMask) = NaN;
        magSqrd                 = self.S ** 2;
        movMeanMagSqrd          = movingMean(magSqrd, 3, axis = 1);
        medMovMeanMagSqrd       = np.median(movMeanMagSqrd, axis = 1);
        medMovMeanMagSqrd       = vectorTranspose(medMovMeanMagSqrd)
        medMovMeanMagSqrdMat    = np.tile(medMovMeanMagSqrd, (1, magSqrd.shape[1]));
        magSqrdMask             = np.ma.masked_where(magSqrd > 10 * medMovMeanMagSqrdMat, magSqrd);
        magSqrd                 = magSqrdMask.filled(np.nan);

        # self.psd = self.dt^2/self.T*mean(magSqrd,2,'omitnan');# use median to exclude outliers from short pulses
        self.psd = self.dt ** 2 / self.T * np.mean(magSqrd, axis = 1, where = ~np.isnan(magSqrd)); # use median to exclude outliers from short pulses
