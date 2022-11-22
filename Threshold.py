from buildTimeCorrelatorMatrix import *
from evfit import *
from wgn import *
import matlab

import numpy as np
import scipy
from math import exp 
from copy import *
from pytictoc import TicToc

def _theFunc(x, *args):
    # theFunc  = @(x) 1 - exp(-exp((-x - mu) / sigma)) - pf;# Equivalent to 1-evcdf(x,mu,sigma)-pf
    return [1 - exp(-exp((-x[0] - args[0]) / args[1])) - args[2]];  # Equivalent to 1-evcdf(x,mu,sigma)-pf

# classdef threshold
class Threshold:
    # properties 
    #     pf              (1,1) double
    #     evMuParam       (1,1) double
    #     evSigmaParam    (1,1) double
    #     thresh1W        (1,1) double
    #     threshVecCoarse (:,1) double
    #     threshVecFine   (:,1) double
    # end

    # function obj = threshold(pf)
    def __init__ (self, pf: float):
        # CONVERSION: using threshold() with no argument is not needed
        # if nargin>0
        #     obj.pf = pf;
        # else
        #     obj.pf = 0.01;
        # end
        # obj.evMuParam        = 0;
        # obj.evSigmaParam     = 0;
        # obj.thresh1W         = 0;  
        # obj.threshVecCoarse  = 0;
        # obj.threshVecFine    = 0;

        self.pf                 = pf
        self.evMuParam          = np.nan
        self.evSigmaParam       = np.nan
        self.thresh1W           = np.nan
        self.threshVecCoarse    = None
        self.threshVecFine      = None

    # function [obj] = setthreshold(obj, WfmCurr, WfmPrev)
    def updateThresholdValuesIfNeeded(self, waveformCurr, waveformPrev):
        #Wq depends on N M J K
        #if old and new N, M, J, K, W, Wf are the same
        #   copy over the fit parameters from prev to curr then 
        #   use the updatepf method to set the others
        #else
        #   use the makenewthreshold method to build out

        # needsUpdate = false;
        # needsUpdate = needsUpdate | (WfmCurr.N ~= WfmPrev.N);
        # needsUpdate = needsUpdate | (WfmCurr.M ~= WfmPrev.M);
        # needsUpdate = needsUpdate | (WfmCurr.J ~= WfmPrev.J);
        # needsUpdate = needsUpdate | (WfmCurr.K ~= WfmPrev.K);
        # needsUpdate = needsUpdate | any(WfmCurr.W ~= WfmPrev.W,'all');
        # needsUpdate = needsUpdate | any(WfmCurr.Wf ~= WfmPrev.Wf,'all');
        needsUpdate = False
        needsUpdate = needsUpdate | (waveformCurr.N != waveformPrev.N);
        needsUpdate = needsUpdate | (waveformCurr.M != waveformPrev.M);
        needsUpdate = needsUpdate | (waveformCurr.J != waveformPrev.J);
        needsUpdate = needsUpdate | (waveformCurr.K != waveformPrev.K);
        needsUpdate = needsUpdate | np.any(np.not_equal(waveformCurr.W, waveformPrev.W));
        needsUpdate = needsUpdate | np.any(np.not_equal(waveformCurr.Wf, waveformPrev.Wf));

        # if needsUpdate
        #     obj = obj.makenewthreshold(WfmCurr);
        # else
        #     obj.evMuParam    = WfmPrev.thresh.evMuParam;
        #     obj.evSigmaParam = WfmPrev.thresh.evSigmaParam;
        #     obj = obj.updatepf(WfmCurr, obj.pf); # Not actually updating the pf, just using the method to set all the other parameters
        # end
        if needsUpdate:
            self.generateThresholdValuesFromWaveform(waveformCurr);
        else:
            self.evMuParam    = waveformPrev.thresh.evMuParam;
            self.evSigmaParam = waveformPrev.thresh.evSigmaParam;
            self = self.updatePf(waveformCurr, self.pf); #Not actually updating the pf, just using the method to set all the other parameters

    def _evthresh(self, evMuParam, evSigmaParam, pf):
        #thresh   = fzero(theFunc,0);# theFunc monitonically decrease, so starting at x = 0 should always converge
        # theFunc monitonically decrease, so starting at x = 0 should always converge
        return scipy.optimize.fsolve(_theFunc, [80000], args = (evMuParam, evSigmaParam, pf))[0];
        
    # function [obj] = updatepf(obj, Wfm, pfNew)
    def updatePf(self, waveform, pfNew):
        # obj.pf = pfNew;
        # thresh = evthresh(obj.evMuParam,obj.evSigmaParam, pfNew); # Build a single threshold value at 1 W bin power
        # obj    = obj.setthreshprops(thresh, Wfm);                   # Set thresholds for each bin based on their bin powers
        self.pf = pfNew
        thresh = self._evthresh(self.evMuParam, self.evSigmaParam, self.pf)           # Build a single threshold value at 1 W bin power
        self._setThreshProps(thresh, waveform);   # Set thresholds for each bin based on their bin powers

    # function [obj] = makenewthreshold(obj, Wfm)
    def generateThresholdValuesFromWaveform(self, Wfm):
        # BUILDTHRESHOLD generates a threshold vector for the waveform argument
        # based on the false alarm probability input.
        # 
        # This function creates a vector of thresholds for the incoherently summed
        # results of the data in the input waveform. The probability of false alarm
        # input is used for threshold generation. The output vector is a spectrally
        # tuned threshold for each frequency in the STFT of the waveform. These
        # thresholds are dependent on the power spectral density for each frequency
        # bin, so areas of the spectrum with a high noise floor will have a higher
        # output thresholde value
        # 
        # INPUTS:
        #   Wfm     A single waveform object
        #   PF      A scalar probability of false alarm value (0 1];
        # OUTPUTS:
        #   newThresh   A vector of threshold with as many elements as rows in the
        #               S matrix of the wfmstft object within the Wfm input.
        # 
        # Author:    Michael W. Shafer
        # Date:      2022-05-04
        # --------------------------------------------------------------------------

        PF = self.pf;

        # This will be the reference power for the trials. Thresholds will be
        # interpolated for each bin from this value based on their bin power
        medPowAllFreqBins = 1; # median(freqBinPow);

        # stftSz     = size(Wfm.stft.S);
        # nTimeWinds = stftSz(2);
        # nFreqBins  = stftSz(1);
        stftSz     = np.shape(Wfm.stft.S);
        nTimeWinds = stftSz[1];
        nFreqBins  = stftSz[0];

        # Build the Wq time correlation matrix
        # Wq = buildtimecorrelatormatrix(Wfm.N, Wfm.M, Wfm.J, Wfm.K);
        # if nTimeWinds ~= size(Wq,1)
        #     error('UAV-RT: Time correlator/selection matrix must have the same number of rows as the number of columns (time windows) in the waveforms STFT matrix.')
        # end
        Wq = buildTimeCorrelatorMatrix(Wfm.N, Wfm.M, Wfm.J, Wfm.K);
        if nTimeWinds != np.shape(Wq)[0]:
            raise Exception('UAV-RT: Time correlator/selection matrix must have the same number of rows as the number of columns (time windows) in the waveforms STFT matrix.')

        # Here we approximated the number of samples of synthetic noise data needed
        # to get the correct number of time windows. We over estimate here and then
        # clip the number of correct windows after the stft operation.
        nSamps = (nTimeWinds+1)*Wfm.n_ws+Wfm.n_ol;# Based on the STFT help file for the number of windows as a function of samples. We add an additional windows worth of samples to ensure we have enough in our STFT output. We'll clip off any excess after the STFT

        # trials       = 100;                             # Number of sets of synthetic noise to generate
        # scores       = zeros(trials,1);                 # Preallocate the scores matrix
        # Psynthall    = medPowAllFreqBins*nFreqBins;     # Calculate the total power in the waveform for all frequency bins. Units are W/bin * # bins = W
        # xsynth       = wgn(nSamps,trials,Psynthall,'linear','complex'); # Generate the synthetic data
        trials       = 100;                             # Number of sets of synthetic noise to generate
        scores       = np.zeros(trials);                # Preallocate the scores matrix
        Psynthall    = medPowAllFreqBins*nFreqBins;     # Calculate the total power in the waveform for all frequency bins. Units are W/bin * # bins = W
        #xsynth       = wgn(nSamps,trials,Psynthall);    # Generate the synthetic data
        
        # Hack city
        rng = np.random.default_rng()
#        noise_power = 0.01 * Wfm.Fs / 2
        noise_power = Psynthall * Wfm.Fs / 2
        noise = rng.normal(scale=np.sqrt(noise_power), size=(trials, nSamps))

        # [Ssynth,~,~] = stft(xsynth,Wfm.Fs,'Window',Wfm.stft.wind,'OverlapLength',Wfm.n_ol,'FFTLength',Wfm.n_w);
        _, _, Ssynth = scipy.signal.stft(
                                noise, 
                                fs              = Wfm.Fs, 
                                window          = Wfm.stft.wind, 
                                noverlap        = Wfm.n_ol, 
                                nperseg         = Wfm.n_w,
                                nfft            = Wfm.n_w,
                                boundary        = None,
                                return_onesided = False)

        #Ssynth(:,nTimeWinds+1:end,:) = [];              # Trim excess so we have the correct number of windows.
        Ssynth = np.delete(Ssynth, Ssynth.shape[2] - nTimeWinds, 2)

        # Preform the incoherent summation using a matrix multiply.
        # Could use pagetimes.m for this, but it isn't supported for code generation

        # for i = 1:trials
        #     scores(i) = max(abs(Wfm.W'*Ssynth(:,:,i)).^2 * Wq, [], 'all'); # 'all' call finds max across all temporal correlation sets and frequency bins just like we do in the dectection code.
        # end

        a = Wfm.W.transpose()
        for i in range(trials):
            # 'all' call finds max across all temporal correlation sets and frequency bins just like we do in the dectection code.
            absValues = np.abs(a @ Ssynth[i, :, :])
            scores[i] = np.amax(absValues ** 2 @ Wq); 

        # Build the distribution for all scores.
        # Old kernel density estimation method
        # [f,xi]   = ksdensity(scores(:),'BoundaryCorrection','reflection','Support','positive');
        # F        = cumtrapz(xi,f);
        # Updated extreme value estimation method
        # xi = linspace(1/2*min(scores),2*max(scores),1000);
        # paramEstsMaxima = evfit(-scores);
        # cdfVals = evcdf(-xi,paramEstsMaxima(1),paramEstsMaxima(2));
        # F = 1 - cdfVals;
        paramEstsMaxima = evfit((-scores).tolist());
        # mu              = paramEstsMaxima[1;
        # sigma           = paramEstsMaxima(2);

        mu              = paramEstsMaxima[0][0];
        sigma           = paramEstsMaxima[1][0];

        threshMedPow    = self._evthresh(mu, sigma, PF);
        
        self.evMuParam       = mu
        self.evSigmaParam    = sigma

        self._setThreshProps(threshMedPow, Wfm);
        
    # methods(Access = protected)
    #     function [obj] = setthreshprops(obj, thresh, Wfm)
    def _setThreshProps(self, thresh, Wfm):        
        freqBinPSD = Wfm.stft.psd; # Extract psd for current waveform. Units are W/Hz
        # freqBinPow = freqBinPSD*(Wfm.stft.f(2)-Wfm.stft.f(1));  # PSD (W/Hz) times bin width (Hz/bin) gives bin total power in (W/bin)
        freqBinPow = freqBinPSD * (Wfm.stft.f[1] - Wfm.stft.f[0]);  # PSD (W/Hz) times bin width (Hz/bin) gives bin total power in (W/bin)

        pow        = 1; # 1W standard

        # powGrid    = [0 pow];
        # threshGrid = [0 thresh];
        # newThresh = interp1(powGrid,threshGrid,freqBinPow,'linear','extrap');
        powGrid    = [0, pow];
        threshGrid = [0, thresh];
        newThresh = matlab.interp1(powGrid, threshGrid, freqBinPow, kind = 'linear', fill_value = 'extrap');

        # Finally,extrapolating the thresholds that are a little beyond the original
        # frequeny range can result in negative thresholds. Here we copy the first
        # and last valid threshold out to the places where the extrapolations would
        # have occured

        # isnanThreshLogic   = isnan(newThresh);
        # firstTrueThreshInd = find(~isnanThreshLogic, 1,'first');
        # lastTrueThreshInd  = find(~isnanThreshLogic, 1,'last');
        isFiniteLogic       = np.isfinite(newThresh);
        firstTrueThreshInd = matlab.find(isFiniteLogic, 1, 'first')
        lastTrueThreshInd  = matlab.find(isFiniteLogic, 1, 'last')

        firstTrueThresh    = newThresh[firstTrueThreshInd];
        lastTrueThresh     = newThresh[lastTrueThreshInd];

        # newThresh(1:firstTrueThreshInd(1))  = firstTrueThresh; # The (1) call is needed by coder, as it doesn't know that the find call above will only produced a scalar output.
        # newThresh(lastTrueThreshInd(1):end) = lastTrueThresh; # The (1) call is needed by coder, as it doesn't know that the find call above will only produced a scalar output.
        newThresh[:firstTrueThreshInd]  = firstTrueThresh
        newThresh[lastTrueThreshInd:]   = lastTrueThresh
        
        # self.thresh1W        = thresh;
        # self.threshVecCoarse = newThresh;
        # self.threshVecFine   = interp1(Wfm.stft.f,double(newThresh),Wfm.Wf,'linear','extrap');
        self.thresh1W        = deepcopy(thresh)
        self.threshVecCoarse = newThresh;
        self.threshVecFine   = matlab.interp1(Wfm.stft.f, newThresh.astype(float), Wfm.Wf, kind = 'linear', fill_value = 'extrap');
