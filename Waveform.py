import PulseStats
import Threshold
import WaveformSTFT
import matlab as ml
from getTemplate import getTemplate
from weightingMatrix import weightingMatrix
from Pulse import Pulse
from buildTimeCorrelatorMatrix import *
from incohSumToeplitz import *
from vecFind import *

import numpy as np
import math
import warnings
from PIL import Image
from copy import *

# classdef waveform < handle 
class Waveform:
    # WAVFORMs are real or complex time series data at a fixed sampling
    # rate. The class contains a number of relevent properties necessary for
    # processing and searching for pulses in the data. 
    # 
    # PROPERTIES:
    #    ps_pre  # Pulse stats object from previous data (priori)
    #    ps_pos  # Pulse stats object after update (posteriori)
    #    x       # Vector of samples
    # 	Fs      # Sample frequency (Hz)
    #    l       # Length of x (numel)
    #    t_0     # Time stamp of first element (s)
    #    t_f     # Time stamp of last element (s)
    # 	t_nextsegstart    Time location where the next segement after this 
    #                      should pick up to ensure sufficient overlap. 
    #                      This is only used when the segment is cleaved to 
    #                      keep track of start and stop locations of each 
    #                      segment.
    # 	stft    A WFMSTFT object of x with properties
    # 	OLF     Overlap fraction
    # 	K       The number of pulses this waveform was built to contain based on its length
    #    W       The weighting matrix
    #    Wf      The weighting matrix frequency vector
    #          
    # 	STFT window related properties
    #        (All can by updated with the updateprioridependentprops	method)
    #    	n_p         Samples for each pulse
    #        n_w         Samples in each window
    #    	n_ol        Samples overlapped in windows
    #        n_ws        Samples for each window step forward
    #    	t_ws        Time for each window step forward (s)
    #        n_ip        Interpulse time in samples
    #    	N           Interpulse time in units of STFT windows
    #        M           Interpulse uncertainty time in units of STFT windows
    #        J           Interpulse jitter time in units of STFT windows
    # 
    # METHODS SUMMARY:
    #    WAVEFORM    Constructs an instance of this class
    #    T           Provides a time vector of x based on t_0 and Fs
    #    ADDDATA     Tacks new data onto the end of the waveform.
    #    CLEAVE      Generates a new waveform with by cutting data from the parent
    #    LEAVE       Generates a new waveform with by copying data from the parent
    #    FINDPULSE   Looks for pulses in the waveform
    #    UPDATE_POSTERIORI updates post. pulsestats based on a pulse list
    #    SETPRIORIDEPENDENTPROPS updates waveform props
    #    GETSNNR     Sets Signal+noise to noise ratio for pulses in pulse list
    #    PROCESS     Processes the waveform for pulses (detection)
    #    SETWEIGHTINGMATRIX  Sets the W and Wf properties per the other
    #                        waveform props
    # ----------------------------------------------------------------------
    # Author: Michael W. Shafer
    # Date: 2020-05-28
    # Change log:
    # 
    # 
    # ----------------------------------------------------------------------
    # # 
#     properties (SetAccess = public)
#         ps_pre (1, 1)       # Pulse stats object from previous data (priori)
#         ps_pos (1, 1)       # Pulse stats object after update (posteriori)
#         K      (1, 1) double# The number of pulses this waveform was built to contain based on its length
#         thresh (1, 1)       # Threshold object
# #      end
# #      properties (SetAccess = protected)
#         x   (1, :) single   # Vector of samples
#         Fs  (1, 1) double   # Sample frequency (Hz)
#         l   (1, 1) double   # Length of x (numel)
#         t_0 (1, 1) double   # Time stamp of first element (s)
#         t_f (1, 1) double   # Time stamp of last element (s)
#         t_nextsegstart (1, 2) double
#         # Time location where the next segement after this should
#         # pick up to ensure sufficient overlap. This is only used
#         # when the segment is cleaved to keep track of start and
#         # stop locations of each segment.
#         stft (1, 1)         # WFMSTFT object of x with properties
#         OLF  (1, 1) double  # Overlap fraction
#         W    (:, :) double  # Spectral weighing matrix
#         Wf   (:, 1) double  # Frequency output of the spectral weighting matrix
#         # #  STFT window related properties
#         # All can by updated with the updateprioridependentprops method
#         n_p  (1, 1) double  # Samples for each pulse
#         n_w  (1, 1) double  # Samples in each window
#         n_ol (1, 1) double  # Samples overlapped in windows
#         n_ws (1, 1) double  # Samples for each window step forward
#         t_ws (1, 1) double  # Time for each window step forward (s)
#         n_ip (1, 1) double  # Interpulse time in samples
#         N    (1, 1) double  # Interpulse time in units of STFT windows
#         M    (1, 1) double  # Interpulse uncertainty time in units of STFT windows
#         J    (1, 1) double  # Interpulse jitter in units of STFT windows
#     end
    
    # methods
    #     function obj = waveform(x, Fs, t_0, ps_pre, OLF, thresh)
    def __init__ (obj, x: np.ndarray, Fs: float, t_0: float, ps_pre: PulseStats, OLF: float, thresh: Threshold):
        if x is not None and x.ndim != 1:
            raise

        obj.x   = x;                    # Data vector
        if x is None:
            obj.l   = 0                 # Elements in the data
            obj.t_f = 0                 # Time stamp of last element
        else:
            obj.l   = len(x);           # Elements in the data
            obj.t_f = t_0+(len(x)-1)/Fs;# Time stamp of last element
        obj.Fs  = Fs;                   # Sample rate
        obj.t_0 = t_0;                  # Time stamp of first element
        obj.t_nextsegstart = t_0;       # This is the time when next
                                        # segment should start to
                                        # ensure sufficient overlap.
                                        # Will need to be updated 
                                        # elsewhere. 
        obj.ps_pre  = ps_pre;
        obj.OLF     = OLF;              # Overlap Fraction for STFT
        obj.K       = 0                 # Unknown number of pulses.
        obj.stft    = None              # Unknown values but set types
        obj.W       = None              
        obj.Wf      = None              
        # obj.ps_pos  = ps_pre.makepropertycopy;  # Current stats are same as previous during construction
        obj.ps_pos  = deepcopy(ps_pre)  # Current stats are same as previous during construction

        obj.setPrioriDependentProps(ps_pre)
        obj.thresh  = thresh;

    def __str__(self): 
        return "Waveform(%x): l:%f t_f:%f Fs:%f t_0:%f t_nextsegstart:%f OLF:%f K:%d n_p:%d n_w:%d n_ol:%d n_ws:%d t_ws:%d n_ip:%d N:%f, K:%d J:%f" % \
            (id(self), self.l, self.t_f, self.Fs, self.t_0, self.t_nextsegstart, self.OLF, self.K, \
                self.n_p, self.n_w, self.n_ol, self.n_ws, self.t_ws, self.n_ip, self.N, self.K, self.J)

    # function t_out = t(obj)
    def t(obj):
        # T provides a time vector of x based on t_0 and Fs
        # 
        # INPUTS:  
        #    obj     The waveform object
        # 
        # OUTPUTS:
        #    t_out   A time vector for obj.x at sample frequency Fs 
        # 
        # # 
        # t_out = obj.t_0+(0:(obj.l-1))/obj.Fs;
        t_out = obj.t_0 + np.arange(obj.l - 1) / obj.Fs;

    # function [] = adddata(obj,x)
    def addData(obj, x: np.ndarray):
        # ADDDATA Tacks new data onto the end of the waveform. This
        # method updates the objects properties that depend on x after
        # appending the new data. 
        # 
        # Note: This method does not compute the STFT of the new
        # data that is appended. This can't be done without knowledge of
        # over which samples any previous STFT was computer, namely
        # where the final STFT window ended. Therefore, it is the
        # responsibility of the caller of this method to also append the
        # objects stft property (the wfmstft object) with the additional
        # STFT columns associated with the added data. See
        # wfmstft.adddata()
        # 
        # INPUTS: 
        #    obj     The waveform object
        #    x       Data to append to the end of obj.
        # OUTPUTS:
        #    none
        # 
        # # 
        # Flatten to be like x of the waveform

        if x.ndim != 1:
            raise

        # obj.x = [obj.x,reshape(x,1,numel(x))];
        # obj.l = length(obj.x);
        # obj.t_f = obj.t_f+length(x)/obj.Fs;
        obj.x.append(x)
        obj.l = len(obj.x);
        obj.t_f = obj.t_f + len(x) / obj.Fs;

#              # Only recompute the STFT if it was already computed
#              if ~isempty(obj.stft.S)
#                  spectro(obj)
#              end

    # function [] = spectro(obj)
    def spectro(obj):
        # SPECTRO Executes an STFT of x and sets up the wfmstst object
        # for the waveform
        # 
        # INPUTS: 
        #    obj     The waveform object
        # OUTPUTS:
        #    none
        # 
        # # 

        # obj.stft = wfmstft(obj);
        obj.stft = WaveformSTFT.WaveformSTFT(obj)

    # function [wfmout] = leave(obj,K,t0,ps_pre)
    def leave(obj, K: float, *args): # args - t0: float, ps_pre: PulseStats
        # LEAVE Grabs enough samples from the beginning of the waveform
        # to ensure they will contain at least K pulses, but not more
        # than K+1. It does not affect the parent object. The start of
        # the samples grabbed begins at t0.
        # 
        # The number of samples used depends on the repetition
        # interval and uncertainty estimates of the previouse puslestats
        # object of the waveform being leaved.
        
        # This method is based on the cleaved method, but eliminates the
        # parts of the code that modify the parent waveform.
        
        # The output waveform has a blank ps_pos property - it doesn't
        # carry any information of the parent's processing into the
        # newly spawned waveform. The new waveform has ps_pre of the
        # input to the method.
        
        # INPUTS:
        #    K   1x1     Pulse wanted in cleaved segment. May get K+1
        #                pulses in segement but not fewer than K.
        #    t0  1x1     The time where the wfmout should start. If t0
        #                is not an integer number of samples away from
        #                the start time of the parent waveform, t0 will
        #                be reset to the next lowest sample (slightly
        #                earlier).
        # OUTPUTS:
        #    wfmout      A waveform object of the cleaved data.
        # 
        # 
        # # 

        # if nargin == 2
        #     t0 = 0;
        #     ps_pre=obj.ps_pre;
        # elseif nargin == 3
        #     ps_pre=obj.ps_pre;
        # end
        argCount = len(args)
        if argCount == 0:
            t0      = 0;
            ps_pre  = obj.ps_pre;
        elif argCount == 1:
            t0      = args[0]
            ps_pre  = obj.ps_pre;
        elif argCount == 2:
            t0      = args[0]
            ps_pre  = args[1]

        t = obj.t;
        # if t0<t(1)
        #     error('Requested time that is before the start time of the parent. Choose a time within the bounds of the waveform time.')
        # end
        # if t0>t(end)
        #     error('Requested time that is later than last time of the parent. Choose a time within the bounds of the waveform time.')
        # end
        if t0 < t[1]:
            raise Exception('Requested time that is before the start time of the parent. Choose a time within the bounds of the waveform time.')
        if t0 > t[-1]:
            raise Exception('Requested time that is later than last time of the parent. Choose a time within the bounds of the waveform time.')
        
        # L = length(obj.x);
        L = len(obj.x);

        # Get relavent variables and rename for readable code.
        [n_p,n_w,n_ol,n_ws,t_ws,n_ip,N,M,J] = obj.getPrioriDependentProps(ps_pre);            
        overlap_windows        = 2*(K*M+J);
        overlap_samps          = n_ws*overlap_windows;# K*M_in_samps;

        # Ideally,
        # windows_in_cleaved_wfm = (K*(N+M)+1);
        # and
        # samples_in_cleaved_wfm = n_ws*windows_in_cleaved_wfm,
        # but the number of columns (time windows) that result from the
        # stft function is dependent on a math.floor operation. As a result
        # we have to solve for the number of samples that accounts for
        # this variablility. See research notebook entry 2020-05-22. The
        # result is,
        # samples_in_cleaved_wfm = n_ws*(K*(N+M)+J+1+1)+n_ol;
        # windows_in_cleaved_wfm = math.floor(samples_in_cleaved_wfm/n_ws);
        # This will produce [K*(N+M)+1+1] windows. The remainder will be
        # from the n_ol. It is okay that we don't include that fraction
        # of a window in windows_in_cleaved_wfm because that variable is
        # only used to transfer a portion of the STFT matrix into the
        # cleaved waveform.
#              if K~=1 # See 2022-04-12 for definitions
#                  samples_in_cleaved_wfm = n_ws*(K*(N+M)-2*M)+n_ol;
#              else
#                  samples_in_cleaved_wfm = n_ws*(N+M+J)+n_ol;
#              end
#             windows_in_cleaved_wfm = math.floor((samples_in_cleaved_wfm-n_ol)/n_ws);
        # See 2022-07-11 for updates to samples def
        samples_in_cleaved_wfm = n_ws*(K*(N+M)+J+1)+n_ol;
        # windows_in_cleaved_wfm = math.floor((samples_in_cleaved_wfm-n_ol)/n_ws);
        windows_in_cleaved_wfm = math.floor((samples_in_cleaved_wfm-n_ol)/n_ws);
                    
        
        # Find the location of where to start
        # ind_start  = find(t<=t0,1,'last');
        ind_start  = ml.find(t <= t0, 1, 'last')[0];
        
        # Figure out what to cut and what to keep
        # inds4cleavedwfm   = (ind_start-1)*ones(1,samples_in_cleaved_wfm)+(1:samples_in_cleaved_wfm);
        # inds4remainingwfm = ind_start-1+samples_in_cleaved_wfm-overlap_samps:L;

        # inds4cleavedwfm   = (ind_start-1)*ones(1,samples_in_cleaved_wfm)+(1:samples_in_cleaved_wfm);
        # inds4remainingwfm = ind_start-1+samples_in_cleaved_wfm-overlap_samps:L;
        inds4cleavedwfm   = (ind_start - 1) * np.ones(samples_in_cleaved_wfm) + np.arange(samples_in_cleaved_wfm);
        inds4remainingwfm = ind_start - 1 + samples_in_cleaved_wfm - np.arange(overlap_samps, L + 1);

        # If we are at the end of the waveform we take all that is there
        # and provide it as the output wavform. Otherwise we clip what
        # we need and update the waveform from which we are cleaving.

        # if inds4cleavedwfm(end)>=L# samples_in_cleaved_wfm>=L# If we are at the end of the waveform
        #     wfmout = waveform(obj.x(inds4cleavedwfm(1):end),obj.Fs,obj.t_0,ps_pre,obj.OLF, obj.thresh);               
        #     # Maintain priori dependent properties.
        #     wfmout.setprioridependentprops(wfmout.ps_pre)
        #     wfmout.t_nextsegstart = obj.t_f+1/obj.Fs;
        #     wfmout.K	= K; # Set number of pulses
        # else
        #     # Create the cleaved waveform segment
        #     x_out   = obj.x(inds4cleavedwfm);
        #     wfmout  = waveform(x_out,obj.Fs,t(inds4cleavedwfm(1)),ps_pre,obj.OLF, obj.thresh);
        #     wfmout.setprioridependentprops(wfmout.ps_pre)
        #     wfmout.t_nextsegstart = t(inds4remainingwfm(1));
        #     wfmout.K    = K;
        if inds4cleavedwfm[-1] >= L: # samples_in_cleaved_wfm>=L# If we are at the end of the waveform
            wfmout = Waveform(obj.x[inds4cleavedwfm[0]:], obj.Fs, obj.t_0, ps_pre, obj.OLF, obj.thresh);               
            # Maintain priori dependent properties.
            wfmout.setPrioriDependentProps(wfmout.ps_pre)
            wfmout.t_nextsegstart   = obj.t_f + 1 / obj.Fs;
            wfmout.K	            = K;
        else:
            # Create the cleaved waveform segment
            x_out   = obj.x[inds4cleavedwfm];
            wfmout  = Waveform(x_out, obj.Fs, t[inds4cleavedwfm[0]], ps_pre, obj.OLF, obj.thresh);
            wfmout.setPrioriDependentProps(wfmout.ps_pre)
            wfmout.t_nextsegstart   = t(inds4remainingwfm(1));
            wfmout.K                = K;
            
            # If stft has been run on this wfm, we cut out the windows
            # for the new waveform and create its wfmstft object. We
            # also update the wfmstft object from which we are cleaving

            # if ~isempty(obj.stft.S)
            if obj.stft.S is not None:
                # This is the time stamp of the stft window that includes t0
                t0_wind = t0 + obj.n_w / obj.Fs / 2;

                # wind_start = find(obj.stft.t<=t0_wind,1,'last');
                # winds4cleavedwfm    = wind_start-1+(1:windows_in_cleaved_wfm);
                wind_start          = ml.find(obj.stft.t <= t0_wind, 1, 'last')[0]
                winds4cleavedwfm    = wind_start - 1 + np.arange(windows_in_cleaved_wfm);

                wfmout.stft         = deepcopy(obj.stft.copy)     # Set to entire object to get all props
                if samples_in_cleaved_wfm >= L: # If we are at the end of the waveform
                    # wfmout.stft.S = wfmout.stft.S(:,winds4cleavedwfm(1):end);   # Clip S for windows of this segment
                    # wfmout.stft.updatepsd();
                    # wfmout.stft.t = wfmout.stft.t(winds4cleavedwfm(1):end);     # Downselect the time vector.
                    wfmout.stft.S = wfmout.stft.S[:, winds4cleavedwfm[0]:]; # Clip S for windows of this segment
                    wfmout.stft.updatePsd();
                    wfmout.stft.t = wfmout.stft.t[winds4cleavedwfm[0]:];    # Downselect the time vector.
                else:
                    # wfmout.stft.S = wfmout.stft.S(:,winds4cleavedwfm);  # Clip S for windows of this segment
                    # wfmout.stft.updatepsd();
                    # wfmout.stft.t = wfmout.stft.t(winds4cleavedwfm);    # Downselect the time vector.
                    wfmout.stft.S = wfmout.stft.S[:, winds4cleavedwfm]; # Clip S for windows of this segment
                    wfmout.stft.updatepsd();
                    wfmout.stft.t = wfmout.stft.t[winds4cleavedwfm];    # Downselect the time vector.
        
    # function [n_p,n_w,n_ol,n_ws,t_ws,n_ip,N,M,J] = getprioridependentprops(obj,ps_obj)
    def getPrioriDependentProps(obj, ps_obj):
        # GETPRIORIDEPENDENTVARS returns the properties in the
        # waveform that are dependent on prior pulse data estimates. It
        # depends on the waveform properties list Fs etc, as well as
        # pulse stats like t_ip, etc.
        # INPUTS:  ps_obj  1x1 pulse stats object
        # OUTPUTS:  pulse stats object

        n_p  = math.ceil(ps_obj.t_p*obj.Fs); # Samples per pulse
        n_w  = 1*n_p;                   # Number of elements in STFT window
        n_ol = math.floor(obj.OLF*n_w);      # Number of elements overlap in STFT window
        n_ws = n_w-n_ol;                # Number of elements in each step of STFT
        t_ws = n_ws/obj.Fs;             # Time of each step of STFT
        
        n_ip  = math.ceil(ps_obj.t_ip * obj.Fs);
        n_ipu = math.ceil(ps_obj.t_ipu * obj.Fs);
        n_ipj = math.ceil(ps_obj.t_ipj*obj.Fs);
        N    = math.floor(n_ip/n_ws);
        M    = math.ceil(n_ipu/n_ws);
        J    = math.ceil(n_ipj/n_ws);

        return [n_p, n_w, n_ol, n_ws, t_ws, n_ip, N, M, J]
        
    # function [] = setprioridependentprops(obj,ps_obj)
    def setPrioriDependentProps(obj, ps_obj):
        # SETPRIORIDEPENDENTVARS updates the properties in the
        # waveform that are dependent on properties in a pulsestats
        # object. 
        # 
        # INPUTS:  
        #    obj     waveform object
        #    ps_obj  a pulse stats object
        # OUTPUTS: 
        #    None (This method modifies the waveform object properties) 
        # 
        # # 
        
        [n_p,n_w,n_ol,n_ws,t_ws,n_ip,N,M,J] = obj.getPrioriDependentProps(ps_obj);
        obj.n_p  = n_p;      # Samples per pulse
        obj.n_w  = n_w;      # Number of elements in STFT window
        obj.n_ol = n_ol;     # Number of elements overlap in STFT window
        obj.n_ws = n_ws;     # Number of elements in each step of STFT
        obj.t_ws = t_ws;     # Time of each step of STFT
        obj.n_ip = n_ip;     # Samples of interpulse duration
        obj.N    = N;        # Baseline interpulse duration in units of STFT windows
        obj.M    = M;        # Interpulse duration deviation from baselines in units of STFT windows
        obj.J    = J;        # Amount of deviation from the PRI means to search
        
        K                 = obj.K;
        overlap_windows   = 2*(K*M+J);
        overlap_samps     = n_ws*overlap_windows;
#              if K ~= 1
#                  samplesforKpulses = n_ws*(K*(N+M)-2*M)+n_ol;
#              else
#                  samplesforKpulses = n_ws*(N+M+J+1)+n_ol;
#              end
        # See 2022-07-11 for updates to samples def
        samplesforKpulses = n_ws*(K*(N+M)+J+1)+n_ol;
        
        # obj.t_nextsegstart  = obj.t_0+(samplesforKpulses)/obj.Fs; # Don't need the overlap here since the next segment starts at samplesforKpulses+n_ol-n_ol from current sample
        obj.t_nextsegstart  = obj.t_0+(samplesforKpulses-overlap_samps)/obj.Fs; # Don't need the overlap here since the next segment starts at samplesforKpulses+n_ol-n_ol from current sample
        
#              if isempty(obj.TimeCorr)
#                  obj.TimeCorr = TemporalCorrelator(N, M, J);
#              else
#                  obj.TimeCorr.update(N, M, J);
#              end

    # function [] = setweightingmatrix(obj, zetas)
    def setWeightingMatrix(obj, zetas):
        # SETWEIGHTINGMATRIX method sets the W and Wf properties of
        # the waveform. These are the weighting matrix and the
        # frequencys (Wf) at which they are computed.
        # INPUTS:
        #    none
        # OUTPUTS:
        #    none other than setting obj.Wf and obj.W
        # ----------------------------------------------------------
        # 
        # Here we build the spectral scaling vector. We make it the same
        # size as the FFT length in the STFT operation, so it has the
        # same frequency resolution.
        # How many frequency bins are there?
        if obj.stft.f is None or len(obj.stft.f) == 0:
            raise Exception('UAV-RT: Weighting matrix requires waveform stft properties to be set. Set the waveform stft property with the spectro method before setting the weighting matrix properties. ')
        fftlength = len(obj.stft.f);
        # Build a pulse time domain template with the same number 
        # of samples as frequency bins:
        w_time_domain = getTemplate(obj,fftlength);
        # Build weighting matrix
        [obj.W, obj.Wf] = weightingMatrix(w_time_domain, obj.Fs,zetas, 'centered')

    # function [] = process(obj,mode,selection_mode,excluded_freq_bands)
    def process(obj, mode, selection_mode, excluded_freq_bands):
        # PROCESS is a method that runs the pulse detection algorithm on
        # a waveform object. 
        # 
        # This method is is intended to be run on a waveform
        # segment that is short enought to contain at most K+1 pulses,
        # but at least K pulses, where K is the number of pulses to be
        # summed. Input paramets allow for the selection of search mode,
        # the focus mode, and the selection mode. The search mode
        # ('mode') is eithr 'discovery', 'confirmation', or 'tracking'
        # and selection of each of these affects how priori information
        # is used to reduce the frequency or time scope of the pulse
        # search. This also affects how the resulting ps_pos property of
        # the waveform is set. The false alarm probabilty is used along
        # with the decision table in order to determine the thresholds
        # used for the decision making on pulse detection. 
        
        # Mode description:
        # Processing can be done in either discovery 'D', confirmation
        # 'C', or tracking 'T' modes. The selection of these modes
        # affects how time and frequency focusing is used to improve
        # processing time. In discovery mode is used for intial
        # detection when little is know about where pulses might exist
        # in time or frequeny. Confirmation is similar to discovery, but
        # is used after an initial detection to confirm that the
        # subsequent pulses exists where they should based on 
        # forecasting from a previous detection. After pulses are
        # confirmed, the mode can be set to tracking which reduces the
        # frequency and time search space to improve processing speed. 
        # 
        # 
        #    Discovey mode:      'D'
        #        This mode uses no priori information and looks at the 
        #        widest possible time bounds for pulses. If there is 
        #        no priori frequency information in the ps_pre property 
        #        or the focus mode is set to 'open', discovery mode 
        #        will search across all frequencies. Otherwise it
        #        will search over all frequencies. If it detects pulses
        #        in the dataset that exceed the decision threshold, the
        #        posterior pulsestats object will be set to a mode of
        #        'C' so that the next segment processed will run in
        #        confirmation mode. If a detection or set of detections
        #        are made, the 'selection_mode' input will dictate how
        #        the posterior pulsestat property is set. 
        # 
        #    Confirmation mode:  'C'
        #        This mode is similar to discovery mode, but after a
        #        detection is made, it will compare the spectral and
        #        temporal location of the detected pulses to determine
        #        if the pulses align with what could be predicted from
        #        the pulses listed in the priori information of the
        #        waveform getting processed. If these pulses are
        #        confirmed, then the posteriori pulsestats mode property
        #        is set to 'T' so that subsequent segments can move to
        #        tracking mode. If pulses are detected but not where
        #        they should have been based on predictions from the
        #        priori, then the posteriori mode is set back to 'D'.
        # 
        #    Tracking mode:      'T'
        #        Tracking mode uses priori pulsestats information in
        #        order to reduce the time and/or frequency space used
        #        for detection. The significantly speed processing
        #        speeds, but can miss detections if pulses move in time
        #        from where they should be based on a previous segment's
        #        detection. If using tracking mode and pulses aren't
        #        detected (scoresV don't meet the decision threhold), the
        #        posteriori pulsestats mode is set back to 'D'.
        # 
        # INPUTS:
        #    mode    The search mode used. Must be string 'D' for
        #            discovery, 'C' for confirmation, or 'T' for
        #            tracking. 
        #    focus_mode  How the algorithm decided whether or not to
        #            focus on a particular pulse. This mode can be set
        #            to 'open' wherein no focusing is done and the
        #            processing always proceeds along in discovery mode
        #            without narrowing the frequency or time span of the
        #            search. If this mode is set to 'focus', the method
        #            will allow for down selection of detected pulses
        #            for subsequent processing. The method of this
        #            downselection is determined by the 'selection_mode'
        #            input. 
        #    selection_mode  How the algorithm narrows the scope for
        #            subsequent segment processing. This input is
        #            used when in 'discovery' mode. If pulses are
        #            detected, the focus mode will dictate which
        #            pulse is focused on for subsequent data
        #            segments. Focus mode can be set to 'most' to
        #            focus on the most powerful detecte pulse or
        #            'least' to focus on the least powerful detected
        #            pulse. See DETECT.M for more
        #            information on these focus modes. 
        #    zetas   A vector of fractions of the DFT frequency bin
        #            width for which to compute and include in the
        #            spectral weighting matrix output. If no interbin
        #            shifting is desired, enter 0. Range of inputs
        #            [0,1). See WEIGHTINGMATRIX.m for more information
        #            on this vector.
        #    PF      A numeric probability of false alarm for decision
        #            threshold determination. Value range of PF must be
        #            within the pf property vector of the decision_table
        #            input. 
        #    decision_table  A structure with two properties - 'pf' and
        #            'values'. The pf property is a vector of false alarm
        #            probabilities in decimal form (not percentage). The
        #            values property is vector of the same size as pf
        #            with the decision thresholds for the corresponding
        #            false alarm probability. 
        #    excluded_freq_bands   nx2    matrix of bands of
        #                                 frequencies to exclude in
        #                                 the search. This is useful if
        #                                 there has already been a tag
        #                                 found and you want to exclude
        #                                 that band when looking for
        #                                 other frequencies. The lower
        #                                 frequecie of the band should
        #                                 be in the first column and
        #                                 the upper should be in the
        #                                 second. This is only used in
        #                                 the naive frequency search
        #                                 mode. Leave empty [] if no
        #                                 exclusion zones are
        #                                 necessary.
        # 
        # OUTPUTS:
        #    None. This method updates the ps_pos property of the
        #    waveform object 'obj'.
        # 
        # # 
        # 
    
    
        # #  CHECK MODE ENTRANCE REQUIREMENTS
        
        # Determine if there is sufficient priori for the operation of each
        # mode. Check to see if they have entries and if they are finite (not Nan or Inf).
        # We just catch that and then set the have_priori flag to false. 

        # have_priori_freq_band = ~isempty(obj.ps_pre.fstart)&...
        #                         ~isempty(obj.ps_pre.fend)&...
        #                         isfinite(obj.ps_pre.fstart) &...
        #                         isfinite(obj.ps_pre.fend);
        
        have_priori_freq_band = obj.ps_pre.fstart is not None and \
                                obj.ps_pre.fend is not None and \
                                math.isfinite(obj.ps_pre.fstart) and \
                                math.isfinite(obj.ps_pre.fend)
        
        # have_priori_time      = ~isempty(obj.ps_pre.pl(end).t_0)&...
        #                         ~isempty(obj.ps_pre.t_p)&...
        #                         ~isempty(obj.ps_pre.t_ip)&...
        #                         ~isempty(obj.ps_pre.t_ipu)&...
        #                         ~isempty(obj.ps_pre.t_ipj)&...
        #                         isfinite(obj.ps_pre.pl(end).t_0)&...
        #                         isfinite(obj.ps_pre.t_p)&...
        #                         isfinite(obj.ps_pre.t_ip)&...
        #                         isfinite(obj.ps_pre.t_ipu)&...
        #                         isfinite(obj.ps_pre.t_ipj);
        
        have_priori_time      = obj.ps_pre.pl is not None and obj.ps_pre.pl[-1].t_0 is not None and \
                                obj.ps_pre.t_p is not None and \
                                obj.ps_pre.t_ip is not None and \
                                obj.ps_pre.t_ipu is not None and \
                                obj.ps_pre.t_ipj is not None and \
                                math.isfinite(obj.ps_pre.pl[-1].t_0) and \
                                math.isfinite(obj.ps_pre.t_p) and \
                                math.isfinite(obj.ps_pre.t_ip) and \
                                math.isfinite(obj.ps_pre.t_ipu) and \
                                math.isfinite(obj.ps_pre.t_ipj)
        
        
        # Enforce entry requirements on the modes so we don't get errors in
        # the 'C' and 'T' modes, as they required priori freq and time info.
        # if strcmp(mode,'T') && ~have_priori_time && ~have_priori_freq_band
        #     # if strcmp(mode,'C'); attemptedmodestring = 'confirmation';end
        #     attemptedmodestring = 'tracking';
        #     if coder.target('MATLAB')
        #         warning(['UAV-RT: Attempted ',attemptedmodestring,' search mode, but priori information is missing frequency and/or time data. Defaulting to discovery mode.'])
        #     end
        #     mode = 'D';
        # end
        if mode == 'T' and not have_priori_time and not have_priori_freq_band:
            attemptedmodestring = 'tracking';
            warnings.warn(['UAV-RT: Attempted ' + attemptedmodestring + ' search mode, but priori information is missing frequency and/or time data. Defaulting to discovery mode.'])
            mode = 'D';
                
        # if strcmp(mode,'C') && ~have_priori_time 
        #     attemptedmodestring = 'confirmation';
        #     if coder.target('MATLAB')
        #         warning(['UAV-RT: Attempted ',attemptedmodestring,' search mode, but priori information is missing frequency and/or time data. Defaulting to discovery mode.'])
        #     end
        #     mode = 'D';
        # end
        if mode == 'C' and not have_priori_time:
            attemptedmodestring = 'confirmation';
            warnings.warn(['UAV-RT: Attempted ' + attemptedmodestring + ' search mode, but priori information is missing frequency and/or time data. Defaulting to discovery mode.'])
            mode = 'D';

        # if strcmp(mode,'I') && ~have_priori_freq_band
        #     attemptedmodestring = 'frequency informed discovery';
        #     if coder.target('MATLAB')
        #         warning(['UAV-RT: Attempted ',attemptedmodestring,' search mode, but priori information is missing frequency and/or time data. Defaulting to discovery mode.'])
        #     end
        #     mode = 'D';
        # end
        if mode == 'I' and not have_priori_freq_band:
            attemptedmodestring = 'frequency informed discovery';
            warnings.warn(['UAV-RT: Attempted ' + attemptedmodestring + ' search mode, but priori information is missing frequency and/or time data. Defaulting to discovery mode.'])
            mode = 'D';

        # switch mode
        #     case 'D'
        #         freq_search_type = 'naive';
        #         time_search_type = 'naive';
        #         runmode          = 'search';
        #     case 'I'
        #         freq_search_type = 'informed';
        #         time_search_type = 'naive';
        #         runmode          = 'search';
        #     case 'C'
        #         freq_search_type = 'naive';
        #         time_search_type = 'naive';
        #         runmode          = 'confirm';
        #     case 'T'
        #         freq_search_type = 'informed';
        #         time_search_type = 'informed';
        #         runmode          = 'track';
        #     otherwise
        #         freq_search_type = 'naive';
        #         time_search_type = 'naive';
        #         runmode          = 'search';
        # end
        if mode == 'D':
            freq_search_type = 'naive';
            time_search_type = 'naive';
            runmode          = 'search';
        elif mode == 'I':
            freq_search_type = 'informed';
            time_search_type = 'naive';
            runmode          = 'search';
        elif mode == 'C':
            freq_search_type = 'naive';
            time_search_type = 'naive';
            runmode          = 'confirm';
        elif mode == 'T':
            freq_search_type = 'informed';
            time_search_type = 'informed';
            runmode          = 'track';
        else:
            freq_search_type = 'naive';
            time_search_type = 'naive';
            runmode          = 'search';

        if runmode == 'search':        
            # Find all the potential pulses in the dataset
            # [candidatelist,msk,pk_ind] = obj.findpulse(time_search_type,freq_search_type,excluded_freq_bands);
            # if  any([candidatelist.det_dec],'all') & any(isnan(pk_ind),'all') # 'all' input required by coder
            #     if coder.target('MATLAB')
            #         warning('UAV-RT: Candidate pulses were detected that exceeded the decision threshold, but no peaks in the scoresV were detected. This is likely because these high scoring pulses existed at the edges of frequency bounds. Try changing the frequency search bounds and reprocessing. ')
            #     end
            # end
            [candidatelist, msk, pk_ind] = obj.findpulse(time_search_type, freq_search_type, excluded_freq_bands);
            anyPeakScoresAboveThreshold = np.isfinite(pk_ind).any()
            if Pulse.pulseArrayAnyDetected(candidatelist) and anyPeakScoresAboveThreshold:
                warnings.warn('UAV-RT: Candidate pulses were detected that exceeded the decision threshold, but no peaks in the scoresV were detected. This is likely because these high scoring pulses existed at the edges of frequency bounds. Try changing the frequency search bounds and reprocessing. ')

            #  Determine which peak to focus depending on the selection mode
            #  Select a peak if we found had at least one detection
            # if ~isnan(pk_ind):
            #     if strcmp(selection_mode,'most') || isempty(selection_mode)
            #         selected_ind = 1;
            #     elseif strcmp(selection_mode,'least')
            #         selected_ind = numel(pk_ind);
            #     end
            #     # Set the pulselist property in the ps_pos based on the
            #     # downselection of pulses
            #     obj.ps_pos.pl = candidatelist(pk_ind(selected_ind),:);
            # else
            #     # If nothing above threshold was found, fill with empty 
            #     # pulse object
            #     obj.ps_pos.pl = makepulsestruc();
            # end
            if anyPeakScoresAboveThreshold:
                if selection_mode is None or len(selection_mode) == 0:
                    raise
                if selection_mode == 'most':
                    selected_ind = 1;
                elif selection_mode == 'least':
                    selected_ind = ml.numel(pk_ind);
                # Set the pulselist property in the ps_pos based on the
                # downselection of pulses
                obj.ps_pos.pl = candidatelist[pk_ind[selected_ind],:];
            else:
                # If nothing above threshold was found, fill with empty 
                # pulse object
                obj.ps_pos.pl = np.array([Pulse()])

            # Record all candidates for posterity
            obj.ps_pos.clst = candidatelist;
            obj.ps_pos.cmsk = msk;
            obj.ps_pos.cpki = pk_ind;

            #  Detection?
    #         if ~isnan(pk_ind)   # Dection was made
    #             for ip = 1:length(obj.ps_pos.pl)
    #                 obj.ps_pos.pl(ip).con_dec = false;
    #             end
    #             #    Update Mode Recommendation -> Confirmation
    #             obj.ps_pos.mode = 'C';
    #         else    # Dection was not made
    #             # False ->
    #             # Just update the mode recommendation to 'S' (search) 
    #             # so we keep an open search
    #             obj.ps_pos.mode = 'S';# 'D';
    #         end
            if anyPeakScoresAboveThreshold:
                # True ->
                # Update confirmation property for each pulse. False 
                # recorded for confirmation property since we are 
                # currently in discovery mode and can't confirm anything
                # yet. 
                for pulse in obj.ps_pos.pl.flat:
                    pulse.con_dec = False
                
    #                      # Only update posteriori if we are focusing. If in open
    #                      # mode, we don't evolve the understanding of pulse
    #                      # timing in the priori. 
    #                      if strcmp(focus_mode,'focus')
    #                      # Calculate & set post. stats (reduced uncertainty)
    #                      # obj.update_posteriori(obj.ps_pos.pl)
    #                      obj.ps_pos.updateposteriori(obj.ps_pre,obj.ps_pos.pl)
    #                      end
                
                #    Update Mode Recommendation -> Confirmation
                obj.ps_pos.mode = 'C';
            else:    # Dection was not made
                # False ->
                # Just update the mode recommendation to 'S' (search) 
                # so we keep an open search
                obj.ps_pos.mode = 'S';# 'D';

            # Set the mode in the pulse and candidate listing for 
            # records. This records the mode that was used in the 
            # detection of the record pulses. This is useful for 
            # debugging. 
            # for tick = 1:numel(obj.ps_pos.pl)
            #     obj.ps_pos.pl(tick).mode = mode;# 'D';
            # end
            # for tick = 1:numel(obj.ps_pos.clst)
            #     obj.ps_pos.clst(tick).mode = mode;#  'D';
            # end
            for pulse in obj.ps_pos.pl.flat:
                pulse.mode = mode;
            for pulse in obj.ps_pos.clst.flat:
                pulse.mode = mode;
        elif runmode == 'confirm':                
            # Find all the potential pulses in the dataset
            [candidatelist, msk, pk_ind] = obj.findpulse(time_search_type, freq_search_type, excluded_freq_bands);
            # if any([candidatelist.det_dec],'all') & any(isnan(pk_ind),'all') # 'all' input required by coder
            #     if coder.target('MATLAB')
            #         warning('UAV-RT: Candidate pulses were detected that exceeded the decision threshold, but no peaks in the scoresV were detected. This is likely because these high scoring pulses existed at the edges of frequency bounds. Try changing the frequency search bounds and reprocessing. ')
            #     end
            # end
            anyPeakScoresAboveThreshold = np.isfinite(pk_ind).any()
            if Pulse.pulseArrayAnyDetected(candidatelist) and anyPeakScoresAboveThreshold:
                warnings.warn('UAV-RT: Candidate pulses were detected that exceeded the decision threshold, but no peaks in the scoresV were detected. This is likely because these high scoring pulses existed at the edges of frequency bounds. Try changing the frequency search bounds and reprocessing. ')

            # At least one pulse group met the threshold
            # if ~isnan(pk_ind)
            #     obj.ps_pos.pl = candidatelist(pk_ind(1),:);
            # else
            #     obj.ps_pos.pl = makepulsestruc();
            # end
            if anyPeakScoresAboveThreshold:
                # Record the detection pulses
                # We only use the highest power pulse group for now
                # because if we are in confirmation mode, we only allow
                # for the selection mode to be 'most'
                obj.ps_pos.pl = candidatelist[pk_ind[0],:]
            else:
                # If nothing above threshold was found, fill with empty 
                # pulse object
                obj.ps_pos.pl = np.array([Pulse()], dtype=object)

            # Record all candidates for posterity
            obj.ps_pos.clst = candidatelist;
            obj.ps_pos.cmsk = msk;
            obj.ps_pos.cpki = pk_ind;
            
            # conflog = false(size(obj.ps_pos)); # Set to all false. Needed
            conflog = np.zeros(obj.ps_pos.pl.size).astype(np.bool_) # Set to all false. Needed

            #  Detection?
            if anyPeakScoresAboveThreshold:
                tref = obj.ps_pre.pl[-1].t_0;
                tp   = obj.ps_pre.t_p;
                tip  = obj.ps_pre.t_ip;
                tipu = obj.ps_pre.t_ipu;
                tipj = obj.ps_pre.t_ipj;

                # if tref>obj.t_0 %First pulse in this segment exists in last segment as well because of overlap
                #     pulsestarttimes_nom        = tref+(tip*(0:(obj.K-1)));
                #     pulsestarttimes_withuncert = tref+((tip-tipu)*(0:(obj.K-1)))-tipj*ones(1,obj.K);
                #     pulseendtimes_withuncert   = tref+((tip+tipu)*(0:(obj.K-1)))+tipj*ones(1,obj.K);
                # else %First pulse in this segment does not exists in last segment as well because of overlap
                #     pulsestarttimes_nom        = tref+(tip*(1:(obj.K)));
                #     pulsestarttimes_withuncert = tref+((tip-tipu)*(1:(obj.K)))-tipj*ones(1,obj.K);
                #     pulseendtimes_withuncert   = tref+((tip+tipu)*(1:(obj.K)))+tipj*ones(1,obj.K);
                # end
                if tref > obj.t_0:
                    # First pulse in this segment exists in last segment as well because of overlap
                    startIndex = 0
                else:
                    # First pulse in this segment does not exists in last segment as well because of overlap
                    startIndex = 1
                pulsestarttimes_nom        = tref + (tip * ml.siVector(startIndex, (obj.K-1 + startIndex)));
                pulsestarttimes_withuncert = tref + ((tip - tipu) * ml.siVector(startIndex, (obj.K-1 + startIndex))) - tipj * np.ones(obj.K);
                pulseendtimes_withuncert   = tref + ((tip + tipu) * ml.siVector(startIndex, (obj.K-1 + startIndex))) + tipj * np.ones(obj.K);

                # Check if pulses started after expected minimum start times
                # minstartlog = [obj.ps_pos.pl(:).t_0]>=pulsestarttimes_withuncert;

                # Check if pulses started before expected maximum start times
                # maxstartlog = [obj.ps_pos.pl(:).t_0]<=pulseendtimes_withuncert;

                plSize = obj.ps_pos.pl.size
                assert plSize == pulsestarttimes_withuncert.size
                t0Vector    = np.empty(plSize, dtype=float)
                for i in range(plSize):
                    t0Vector[i] = obj.ps_pos.pl[i].t_0

                minstartlog = t0Vector >= pulsestarttimes_withuncert
                maxstartlog = t0Vector <= pulsestarttimes_withuncert
                
                # Frequency check. Within +/- 100 Hz of previously
                # detected pulses?
                # freqInBand = [obj.ps_pos.pl.fp] >= [obj.ps_pre.pl.fp]-100 & [obj.ps_pos.pl.fp] <= [obj.ps_pre.pl.fp]+100;
                plSize = obj.ps_pos.pl.size
                fpVector    = np.empty(plSize, dtype=float)
                for i in range(plSize):
                    fpVector[i] = obj.ps_pos.pl[i].fp
                freqInBand = np.logical_and(np.greater_equal(fpVector, fpVector-100),  np.less_equal(fpVector, fpVector+100))

                conflog = maxstartlog & minstartlog & freqInBand;
                # if any(conflog,'all')
                #     #  	Confirmed?
                #     #  		True -> Confirmation = True
                #     conf = true;
                # else
                #     #  		False -> Confirmation = False
                #     conf = false;
                # end
                conf = np.any(conflog)
            else:
                # No peak scoresV detected
                conf = False

            #  Confirmation?
            if conf:
                # Update confirmation property for each pulse
                # for ip = 1:length(obj.ps_pos.pl)
                #     # obj.ps_pos.pl(ip).con_dec = true;
                #     obj.ps_pos.pl(ip).con_dec = conflog(ip);
                # end
                for ip in range(obj.ps_pos.pl.size):
                    obj.ps_pos.pl[ip].con_dec = conflog[ip];
                
#                      # Open focus mode will never get to confirmation, so we
#                      # don't need the 'if' statement here checking the focus
#                      # most like in the discovery case above
#                      #    Calculate & set post. stats (reduced uncertainty)
#                      # obj.update_posteriori(obj.ps_pos.pl)# (Note this records pulse list)
#                      obj.ps_pos.updateposteriori(obj.ps_pre,obj.ps_pos.pl)

                # Update mode suggestion for next segment processing
                #    Mode -> Tracking
                obj.ps_pos.mode = 'T';
            else:
                # False ->
                # Update mode suggestion for next segment processing
                # 	Mode -> Discovery
                obj.ps_pos.mode = 'S';
                # obj.ps_pos.updateposteriori(obj.ps_pre,[]); # No pulses to update the posteriori

            # Set the mode in the pulse and candidate listing for 
            # records. This records the mode that was used in the 
            # detection of the record pulses. This is useful for 
            # debugging. 
            # for tick = 1:numel(obj.ps_pos.pl)
            #     obj.ps_pos.pl(tick).mode = mode;# 'C';
            # end
            # for tick = 1:numel(obj.ps_pos.clst)
            #     obj.ps_pos.clst(tick).mode = mode;# 'C';
            # end
            for pulse in obj.ps_pos.pl.flat:
                pulse.mode = mode;
            for pulse in obj.ps_pos.clst.flat:
                pulse.mode = mode;
        elif runmode == 'track':
            # Find all the potential pulses in the dataset
            [candidatelist, msk, pk_ind] = obj.findpulse(time_search_type, freq_search_type, excluded_freq_bands); 
            # if any([candidatelist.det_dec],'all') & any(isnan(pk_ind),'all') # 'all' input required by coder
            #     if coder.target('MATLAB')
            #         warning('UAV-RT: Candidate pulses were detected that exceeded the decision threshold, but no peaks in the scoresV were detected. This is likely because these high scoring pulses existed at the edges of frequency bounds. Try changing the frequency search bounds and reprocessing. ')
            #     end
            # end
            anyPeakScoresAboveThreshold = np.isfinite(pk_ind).any()
            if any([candidatelist.det_dec],'all') and anyPeakScoresAboveThreshold:
                warnings.warn('UAV-RT: Candidate pulses were detected that exceeded the decision threshold, but no peaks in the scoresV were detected. This is likely because these high scoring pulses existed at the edges of frequency bounds. Try changing the frequency search bounds and reprocessing. ')

            # At least one pulse group met the threshold
            # if anyPeakScoresAboveThreshold:
            #     obj.ps_pos.pl = candidatelist(pk_ind(1),:);
            # else # Nothing met the threshold for detection
            #     obj.ps_pos.pl = makepulsestruc();
            # end
            if anyPeakScoresAboveThreshold:
                # Record the detection pulses
                # We only use the highest power pulse group for now
                # because if we are in confirmation mode, we only allow
                # for the selection mode to be 'most'
                obj.ps_pos.pl = candidatelist[pk_ind[0], :];
            else:
                # Nothing met the threshold for detection
                obj.ps_pos.pl = Pulse()

            obj.ps_pos.clst = candidatelist;
            obj.ps_pos.cmsk = msk;
            obj.ps_pos.cpki = pk_ind;

            #  Detection?
            if anyPeakScoresAboveThreshold:
                # Update confirmation property for each pulse
                # If we are in tracking mode, we have narrowed the time and
                # frequeny bounds, so if there is a detection then we are
                # confirming the previous projections.
                # for ip = 1:length(obj.ps_pos.pl)
                #     obj.ps_pos.pl(ip).con_dec = true;
                # end
                for ip in range(len(obj.ps_pos.pl)):
                    obj.ps_pos.pl[ip].con_dec = True;
                
#                      # Open focus mode will never get to tracking, so we
#                      # don't need the 'if' statement here checking the focuse
#                      # most like in the discovery case above
#                      #    Calculate & set post. Stats (reduced uncertainty)
#                      # obj.update_posteriori(obj.ps_pos.pl)
#                      obj.ps_pos.updateposteriori(obj.ps_pre,obj.ps_pos.pl)

                #    Mode -> Tracking
                # Update mode suggestion for next segment processing
                obj.ps_pos.mode = mode;# 'T';
            else:
                # Update confirmation property for each pulse. Don't need to
                # do this since there are no pulses to record a confirmation
                # decision on.
                
                # Update mode suggestion for next segment processing
                #    Mode -> Discovery
                obj.ps_pos.mode = 'S';# 'D';

            # Set the mode in the pulse and candidate listing for 
            # records. This records the mode that was used in the 
            # detection of the record pulses. This is useful for 
            # debugging. 
            # for tick = 1:numel(obj.ps_pos.pl)
            #     obj.ps_pos.pl(tick).mode    = mode;#  'T';
            # end
            # for tick = 1:numel(obj.ps_pos.clst)
            #     obj.ps_pos.clst(tick).mode  = mode;#  'T';
            # end
            for pulse in obj.ps_pos.pl.flat:
                pulse.mode    = mode;#  'T';
            for pulse in obj.ps_pos.clst.flat:
                pulse.mode  = mode;#  'T';
        else:
            # #  SOMETHING BROKE
            raise Exception('No mode selected')

    # function [pl_out,indiv_msk,peak_ind] = findpulse(obj,time_searchtype,freq_searchtype,excluded_freq_bands_in)
    def findpulse(obj, time_searchtype, freq_searchtype, excluded_freq_bands):
        # FINDPULSE Looks for pulses in the waveform based on its pulse
        # statistics object
        # 
        # This method implemets the incoherent summation algorithm for
        # the waveform. The number of pulses summed (blocks considere)
        # is defined based on the length of obj.x and its priori
        # pulsestats object. When the waveform object is created, the
        # length of data (x property) included should be defined to be
        # long enough so that it is assure to contain no fewer than K
        # pulses. This method doesn't know K, but will create as many
        # blocks as it can based on the priori information. 
        # 
        # Note: The algorithm used for pulse peak energy detection
        # requires that the center frequency of the pulse not be at the
        # edges of the frequency bins in Wf. There needs to be rolloff
        # in signal power above and below the pulse center frequency.
        # Pulses at or just above/below the upper/lower limits of Wf may
        # exceeed the threshold for detection, but not get recorded in
        # the pl_out because the peeling algorithm can't catch them. If
        # this happens, the frequencies used in the priori to describe
        # the bounds of the pulse frequencies should be adjusted or
        # widened to ensure the pulse's spectral power has rolloff above
        # and below the center frequency and that this rolloff is within
        # the bounds of the Wf input. 
        # 
        # 
        # INPUTS:
        #     obj            Waveform object
        #     W              spectral weighting matrix
        #     Wf             frequencies list from baseband of
        #                    corresponding to each row of W
        #     time_searchtype     'naive' or 'informed'
        #     freq_searchtype     'naive' or 'informed'
        #     excluded_freq_bands_in   nx2    matrix of bands of
        #                                  frequencies to exclude in
        #                                  the search. This is useful if
        #                                  there has already been a tag
        #                                  found and you want to exclude
        #                                  that band when looking for
        #                                  other frequencies. The lower
        #                                  frequecies of the band should
        #                                  be in the first column and
        #                                  the upper should be in the
        #                                  second. Leave empty [] if no
        #                                  exclusion zones are
        #                                  necessary. These frequencies
        #                                  are used regardless of the
        #                                  freq_searchtype setting. 
        # OUPUTS:
        # 	pl_out      A matrix of all potential pulses. This will 
        #                have as many rows as Wf elements and as many 
        #                rows as blocks that were possible given the 
        #                length of the data in obj. The number of pulses 
        #                to be searched for would be defined at the 
        #                time of the creation of obj and used to define 
        #                the length of its data. 
        # 
        #    indiv_msk   A logical matrix who's columns identify the
        #                peaks and sideband locations in the spectrum of
        #                each peak. The ith column of indiv_msk
        #                identifies the spectral location of the ith
        #                most powerful pulse group identified. 
        #    peak_ind    A vector of the frequency indicies (of Wf)
        #                where the power of the pulses were found.
        #                These are the row indicies of pl_out where the
        #                peaks exist.
        #                These are the center frequency of the pulse 
        #                where the most energy was detected. The 
        #                sidebands of the ith peak listed in peak_ind is
        #                identified in the ith column of indiv_msk
        # 
        # # 

        # if (size(excluded_freq_bands_in,2)~=2) && (~isempty(excluded_freq_bands_in))
        #     error('Excluded frequency band listing must be a nx2 matrix or empty.')
        # end        
        # if numel(obj.ps_pre)>1
        #     error('findpulse method only supports waveforms with a single priori pulse states object in ps_pre (not a vector)')
        # end
        # if isempty(excluded_freq_bands_in)
        #     excluded_freq_bands = [inf -inf];
        # else
        #     excluded_freq_bands = excluded_freq_bands_in;
        # end    
        # FIXME: HACK
        excluded_freq_bands = np.array([np.inf -np.inf])
        # if excluded_freq_bands is None or ml.size(excluded_freq_bands, 2) != 2:
        #     raise Exception('Excluded frequency band listing must be a nx2 matrix or empty.')
        if isinstance(obj.ps_pre, Pulse):
            raise Exception('findpulse method only supports waveforms with a single priori pulse states object in ps_pre (not a vector)')
        
        # Get relavent variables and rename for readable code.
        N = obj.N;
        M = obj.M;
        J = obj.J;
                    
        # Define naive search window for pulse 1
        naive_wind_start = 1;
        naive_wind_end   = N+M+J; 
        
        # The caller of this method should have already set the
        # threshold in the posteriori pulse stats. We just rename it
        # here to simplify code.
        # thresh = obj.ps_pos.thresh;
        # The caller of this method should have already set the
        # threshold. We just rename it
        # here to simplify code.
        thresh = obj.thresh.threshVecFine;

        
        # if strcmp(time_searchtype,'naive')
        if time_searchtype == 'naive':
            # NAIVE SEARCH
            wind_start = naive_wind_start;
            wind_end   = naive_wind_end;
        # elseif strcmp(time_searchtype,'informed') && isempty(obj.ps_pre.pl(1).t_0)
        elif time_searchtype == 'informed' and obj.ps_pre.pl[1].t_0 is None:
            # INFORMED SEARCH BUT NOT PRIORI FOR START TIME
            warnings.warn('Requested informed search, but previous segment did not have at least one pulse with a recorded start time. Using naive search method.')
            wind_start = naive_wind_start;
            wind_end   = naive_wind_end;
        else:
            # INFORMED SEARCH WITH PRIORI FOR START TIME    
            # Due to segment lengths and overlapping, there can
            # sometimes be more than the desired number of pulses in a
            # segment. One more. If for example we were looking for 3
            # pulses, but the previous segment contained 4, the 
            # algorithm may have localized the latter 3 rather than the
            # first 3. Here we set up the time window search range. 
            # If the last known pulse is in the first part of the time 
            # range of this segement currently getting processed, we 
            # search around that very narrow region. If it
            # isn't (doesn't existing in this segement, only the 
            # previous) we project the last known pulse forward one
            # interpulse time, and develop search region in accordance
            # with the uncertainty of the subsequent pulse.
            
            # Project out search range for first pulse in this segment
            # based on previous pulses.
            # t_lastknown = obj.ps_pre.pl(end).t_0;
            t_lastknown = obj.ps_pre.pl[-1].t_0;
            tp_temp     = obj.ps_pre.t_p;
            # If the last pulse identified occured in this segement, use
            # it to downselect the first pulse search range.
            # Time steps in the STFT windows:
            # stft_dt     = obj.stft.t(2)- obj.stft.t(1); 
            stft_dt     = obj.stft.t[1]- obj.stft.t[0]; 
            
            # IF LAST SEGMENT'S LAST PULSE ALSO LIVES IN THIS SEGMENT
            # BECAUSE OF OVERLAP:
            timetol = obj.n_ws / obj.Fs;
            # if t_lastknown>=obj.stft.t(1)-stft_dt/2 
            if t_lastknown >= obj.stft.t[0]-stft_dt / 2:
                # The stft_dt/2 is because the stft time stamps are for
                #  the middle of each time windpw
                # wind_start = find(obj.stft.t-stft_dt/2 >=t_lastknown,1,'first');
                # wind_end   = find(obj.stft.t-stft_dt/2 <=t_lastknown+tp_temp,1,'last');
                
                # The logic below executes the same logic as the two
                # lines above (left commented for reference) but avoids
                # issues associated when comparing floating point
                # numbers. There can be small errors that make two
                # numbers not equal each other and this addresses that
                # problem. See the Matlab help file on eq.m under the
                # section 'Compare Floating-Point Numbers'.
                # wind_start = find(abs(obj.stft.t-stft_dt/2 - t_lastknown) <=timetol,1,'first');
                # wind_end   = find(abs(obj.stft.t-stft_dt/2 - (t_lastknown+tp_temp))<=timetol,1,'last');
                wind_start = ml.find(np.absolute(obj.stft.t -stft_dt /2 - t_lastknown) <= timetol, 1,'first')[0]
                wind_end   = ml.find(np.absolute(obj.stft.t -stft_dt /2 - (t_lastknown+tp_temp)) <= timetol, 1, 'last')[0]
                
            else:
                # IF LAST SEGMENT'S LAST PULSE DOESN'T LIVE IN THIS SEGMENT:
                # Project forward one pulse in time with
                # +/-2M uncertainty in search range.
                tip_temp = obj.ps_pre.t_ip;
                tipu_temp = obj.ps_pre.t_ipu;
                next_pulse_start = t_lastknown + tip_temp;   # +tp_temp/2;
                # These are the times of the START OF THE PULSE...not 
                # the center. This is why we have stft_dt/2 terms in 
                # subsequent equations.
                t_srch_rng = tipu_temp * [-1, 1] + next_pulse_start; 
                # if t_srch_rng(1)<obj.stft.t(1)-stft_dt/2
                if t_srch_rng[0] < obj.stft.t[0] - stft_dt / 2:
                    # If for some reason the last known pulse is more
                    # that one pulse repetition interval away from the
                    # current waveform start time, use the naive search
                    # range.
                    warnings.warn('Requested informed search, but the last know pulse is more that one interpulse time away from the current segment start time. This means the algorithm would have to project forward by more than one pulse repetition interval. This is not supported. The naive search method will be used.')
                    wind_start = naive_wind_start;
                    wind_end   = naive_wind_end;
                else:
                    # wind_start = find(obj.stft.t-stft_dt/2>=t_srch_rng(1),1,'first');
                    # wind_end   = find(obj.stft.t-stft_dt/2<=t_srch_rng(2),1,'last');
                    # The logic below executes the same logic as the two
                    # lines above (left commented for reference) but 
                    # avoids issues associated when comparing floating 
                    # point numbers. There can be small errors that
                    # make two numbers not equal each other and this 
                    # addresses that problem. See the Matlab help file
                    # on eq.m under the section 'Compare Floating-Point
                    # Numbers'.
                    # wind_start = find(abs(obj.stft.t-stft_dt/2-t_srch_rng(1))<=timetol,1,'first');
                    # wind_end   = find(abs(obj.stft.t-stft_dt/2-t_srch_rng(2))<=timetol,1,'last');
                    wind_start = ml.find(np.absolute(obj.stft.t -stft_dt /2 -t_srch_rng[0]) <= timetol, 1, 'first')[0]
                    wind_end   = ml.find(np.absolute(obj.stft.t -stft_dt /2 -t_srch_rng[1]) <= timetol, 1, 'last')[0]
        
        # time_search_range = [wind_start,wind_end];
        # Build a time search range matrix with one row for each pulse
        # The first column is the first window where that pulse might
        # live in the S matrix, and the second column is the last window
        # wher that pulse might live in the S matrix.

        # timeSearchRange = zeros(obj.K,2);
        # timeSearchRange(1,:) = [wind_start,wind_end];
        # for i = 2:obj.K
        #     timeSearchRange(i,:) = timeSearchRange(1,:)+...
        #                             [(i-1)*(N-M)-J,(i-1)*(N+M)+J];
        # end
        # maxStftTimeWind = size(obj.stft.S,2);
        timeSearchRangeIndices = np.zeros((obj.K, 2));
        timeSearchRangeIndices[0,:] = [wind_start-1, wind_end-1];
        for i in range(1, obj.K):
            timeSearchRangeIndices[i,:] = timeSearchRangeIndices[0, :] + [i * (N-M) - J, i * (N+M) + J];
        maxStftTimeWind = obj.stft.S.shape[1]

        # Limit the search to ranges of time that will be in the sftf
        # matrix

        # timeSearchRange(timeSearchRange>maxStftTimeWind) = maxStftTimeWind;
        # timeSearchRange(timeSearchRange<1) = 1;
        # timeBlinderVec = zeros(maxStftTimeWind,1);
        # for i = 1:obj.K
        #     timeBlinderVec(timeSearchRange(i,1):timeSearchRange(i,2)) = 1;
        # end        
        # if isempty(excluded_freq_bands)
        #     excluded_freq_bands = [inf -inf];
        # end
        timeSearchRangeIndices[timeSearchRangeIndices > maxStftTimeWind-1]  = maxStftTimeWind-1;
        timeSearchRangeIndices[timeSearchRangeIndices < 1]                = 1;
        timeBlinderVec = np.zeros(maxStftTimeWind);
        for i in range(obj.K):
            timeBlinderVec[ml.siVector(timeSearchRangeIndices[i,0], timeSearchRangeIndices[i,1])] = 1;
        
        # Build the excluded frequency mask
        # False where we want to exclude

        # excld_msk_mtrx = ~(vecfind(obj.Wf,'>=',excluded_freq_bands(:,1)) &...
        #                     vecfind(obj.Wf,'<=',excluded_freq_bands(:,2)));
        # excld_msk_vec  = all(excld_msk_mtrx,2);   
        
        #FIXME: Hacked out
        # excld_msk_mtrx = np.logical_not(np.logical_and(vecFind(obj.Wf, '>=', excluded_freq_bands[:, 0]), vecFind(obj.Wf, '<=', excluded_freq_bands[:, 1])))
        # excld_msk_vec  = np.nonzero(excld_msk_mtrx)[1];   
        
        # Build the priori frequency mask
        # Naive or informed frequency search
        if freq_searchtype == 'informed':
            f_lo = obj.ps_pre.fstart;
            f_hi = obj.ps_pre.fend;
            
            # Check to makesure the informed search range is covered by
            # the frequencies available. If the prev bw is listed
            # as NaN, this will fail and will default to the naive
            # search. Freq_mask variable is a logicial vector of the
            # same size as Wf indicating which frequencies to look at.
            
            # if (f_lo<min(obj.Wf,[],'all')) || (f_hi>max(obj.Wf,[],'all'))# isnan(obj.ps_pre.fc) # Naive
            if (f_lo < np.min(obj.Wf)) or (f_hi > np.max(obj.Wf)):
                # IF FREQS ARE UNAVILABLE, USE NAIVE
                warnings.warn('UAVRT:searchtype Requested informed search, but previous segment does not contain a start and/or stop frequency, or those values produces a search range outside the range of frequencies available. Using naive frequency search. Defaulting to naive frequency search.')
            else:
                # IF FREQS ARE AVILABLE, USE INFORMED
                freq_start = ml.find(obj.Wf >= obj.ps_pre.fstart,   1,'first')[0]
                freq_end   = ml.find(obj.Wf <= obj.ps_pre.fend,     1,'last')[0]
            
            freq_ind_rng = [freq_start, freq_end];
            # freq_mask    =  false(size(obj.Wf));
            # freq_mask(freq_ind_rng(1):freq_ind_rng(2))=true;
            freq_mask    =  np.zeros(obj.Wf.shape).astype(np.bool_)
            freq_mask[ml.siVector(freq_ind_rng[0], freq_ind_rng[1])] = True;
        else: 
            # Naive frequency search
            # freq_mask    =  true(size(obj.Wf));
            freq_mask    =  np.ones(obj.Wf.shape).astype(np.bool_)
        
        # If using informed search and excluded frequencies overlap with
        # priori frequencies, warn the user.

        # if strcmp(freq_searchtype,'informed') & any(freq_mask & ~excld_msk_vec ,'all')
        if freq_searchtype == 'informed' and np.any(freq_mask & excld_msk_vec.logical_not()):
            warnings.warn('UAV-RT: Using informed frequency search method and excluded frequencies. Some or all of the priori frequency band used for the informed search overlaps with frequencies specified for exclusion from the search and thus will not be included.')
                    
        # Combine the masks
        #FIXME: Hacked out
        # freq_mask = freq_mask & excld_msk_vec;
        
        # Check that we actually have something to search
        # if ~any(freq_mask,'all')
        if not np.any(freq_mask):
            raise Exception('UAV-RT: Informed frequencies bands and/or excluded frequencies have created a scenario where no frequencies are being considered in the search. This is likely a result of informed frequeny band overlapping with one or more of the excluded frequency bands. Consider reducing the bandwidth of the excluded frequencies.')
        
        # Determine the number of blocks to cut the data into. Should
        # always be the number of pulses currently looking for. 

        # n_blks = floor((size(obj.stft.S,2)-1)/(N+M));
        n_blks = math.floor((obj.stft.S.shape[1] - 1) / (N+M));
        if n_blks == 0:
            raise Exception('UAVRT:insufficientdata: Insufficient samples to be sure even one pulse is in segment')
        
        Wq = buildTimeCorrelatorMatrix(N, M, J, obj.K);
        # [yw_max_all_freq,S_cols] = incohsumtoeplitz(freq_mask,obj.W',obj.stft.S,timeBlinderVec, Wq);# obj.TimeCorr.Wq(obj.K));
        [yw_max_all_freq, S_cols] = incohSumToeplitz(freq_mask, obj.W.conj().transpose(), obj.stft.S, timeBlinderVec, Wq)

# Only pass a freq_msk subset of the Sw matrix to the incohsum
# function. Run the algorithm            
#              tic
#              [yw_max_all_freq,S_cols,S_rows] = incohsumfast(Sw(freq_mask,:),...
#                                                          N,M,...
#                                                          time_search_range,...
#                                                          n_blks);
#              time(1) = toc;
#              tic
#              [yw_max_all_freq,S_cols,S_rows] = incohsum(Sw(freq_mask,:),...
#                                                          N,M,...
#                                                          time_search_range,...
#                                                          n_blks);
#              time(2) = toc;
#              time
#              yw_max_all_freq2 = NaN(numel(Wf),n_blks);
#              yw_max_all_freq2(freq_mask,:) = yw_max_all_freq;
#              yw_max_all_freq = yw_max_all_freq2;                         
#              S_cols2 = NaN(numel(Wf),n_blks);
#              S_cols2(freq_mask,:) = S_cols;
#              S_cols = S_cols2;
        
        # PAPER EQUATION 14
        # y_vec_w = Sw(sub2ind(size(Sw),out(:,1),out(:,2)));
        #  # PAPER EQUATION pre-29
        # gamma   = abs(y_vec_w);
        # # PAPER EQUATION 29
        # z       = sum(gamma.^2);
        
        
        # #  PEELING ALGORITHM
        #  The following algorithm is used to help tease out the central
        #  peaks of the outputs of filter and sort out those entries
        #  that are sideband. It looks through all of the scoresV 
        #  (yw_max_all_freqs), beginning with the highest score. It
        #  looks for other scoresV that are correlated in time and nearby
        #  frequency and identifies them as sidebands. Once the highest
        #  peak's sidebands are identified and excluded from further
        #  consideration as a separate pulse, the algorithm moves to the
        #  next highest score, repeating the sideband identification.
        #  This continues until no more scoresV that exceed the decision
        #  threshold remain. 
        
        # Number of pulses identified for each frequency. This should be
        # equal to the K pulses of the search or n_blks. 
        n_pls    = ml.size(yw_max_all_freq, 2);
        n_freqs  = ml.size(yw_max_all_freq, 1);
        # Calculate the sum of the yw values - the score/test statistic
        scoresV   = ml.sum(yw_max_all_freq, 2)
        # Calculate an estimate of the noise only power in each bin
        # based on the PSD
        # psdAtZetas = interp1(obj.stft.f,obj.stft.psd,Wf,'linear','extrap');
        # freqBinPowAtZetas = psdAtZetas*(Wf(2)-Wf(1));
        [nRowsOfS, nColsOfS] = obj.stft.S.shape
        # sIndsOfBins = sub2ind([nRowsOfS nColsOfS],repmat((1:nRowsOfS)',1,nColsOfS),S_cols);
        nZetas = ml.numel(obj.Wf) // nRowsOfS;
        # sIndsOfBins = sub2ind([nZetas*nRowsOfS nColsOfS],transpose(1:numel(Wf)),S_cols);
        # The few lines below finds the noisePSD, but excludes regions
        # in time and frequency around possible detection. We do
        # detections as the zeta steps, but use the S matrix for PSD
        # estimates so there is a size mismatch that necessitates the
        # code below. 
        # weightedSRowSubs = transpose(1:numel(obj.Wf));
        weightedSRowSubs = ml.transposeV(ml.siVector(0, ml.numel(obj.Wf)-1));
        weightedSRowSubsMat = ml.repmat(weightedSRowSubs, 1, obj.K);
        # sub2ind doesn't support NaN values, so we focus here on those
        # that don't have NaN

        # S_cols_withoutNan               = S_cols(~isnan(S_cols));
        # weightedSRowSubsMat_withoutNan  = weightedSRowSubsMat(~isnan(S_cols));
        # FIXME: Removing ignored not support
        S_cols_notIgnored               = S_cols
        weightedSRowSubsMat_notIgnored  = weightedSRowSubsMat

        # indsOfBins = sub2ind([nZetas*nRowsOfS, nColsOfS], weightedSRowSubsMat_withoutNan, S_cols_withoutNan);
        # indsOfBinsValid = indsOfBins(~isnan(indsOfBins));

        # binMaskMatrix will be a matrix of NaN at potential pulse locations

        # binMaskMatrix = zeros([nZetas*nRowsOfS nColsOfS]);
        # binMaskMatrix(indsOfBinsValid) =  NaN;# 0;
        binMaskMatrix = np.zeros((nZetas*nRowsOfS, nColsOfS))
        binMaskMatrix[weightedSRowSubsMat_notIgnored, S_cols_notIgnored] = np.nan

        # Now add some buffer around the potential pulse locations
        freqShiftedBinMaskMatrix = binMaskMatrix + \
                                    ml.circshift(binMaskMatrix, 1, 1) + \
                                    ml.circshift(binMaskMatrix,-1, 1) + \
                                    ml.circshift(binMaskMatrix, 2, 1) + \
                                    ml.circshift(binMaskMatrix,-2, 1)
        freqtimeShiftedBinMaskMatrix = freqShiftedBinMaskMatrix + \
                                    ml.circshift(freqShiftedBinMaskMatrix, 1, 2) + \
                                    ml.circshift(freqShiftedBinMaskMatrix,-1, 2)

        # freqtimeShiftedBinMaskMatrixScaled = imresize(freqtimeShiftedBinMaskMatrix, [nRowsOfS, nColsOfS]);
        im = Image.fromarray(freqtimeShiftedBinMaskMatrix, mode='F')
        freqtimeShiftedBinMaskMatrixScaled = im.resize((nColsOfS, nRowsOfS))

        dt = 1/obj.Fs;
        T = obj.n_w/obj.Fs;

        # noisePSD = dt^2/T*mean(abs(obj.stft.S+freqtimeShiftedBinMaskMatrixScaled).^2,2,'omitnan');# Add since it is 0 where we expect noise and NaN where there might be a pulse
        abs1 = np.absolute(obj.stft.S + freqtimeShiftedBinMaskMatrixScaled)
        noisePSD = dt**2 / T * np.nanmean(abs1 ** 2, axis=1);

        noisePSDAtZetas = ml.transposeV(ml.interp1(obj.stft.f, noisePSD, obj.Wf, kind='linear', fill_value='extrap'))
        noisePSDAtZetas[noisePSDAtZetas < 0] = 0;
        # fBinWidthZetas = obj.stft.f(2)-obj.stft.f(1);# Not the delta f of the Wf vector, because the frequency bins are the same width, just with half bin steps # 
        fBinWidthZetas = obj.stft.f[1]-obj.stft.f[0] # Not the delta f of the Wf vector, because the frequency bins are the same width, just with half bin steps # 
        
        # Calculate the power at each of the S locations that were
        # selected by the incohsum function. scoresV are the mag squared
        # S values. Mult by dt^2/T to get psd, then mult by the width of
        # the frequency bin to get power. We have to do this because the
        # psd in the stft object uses dt^2/T factor for the psd calc. 
        signalPlusNoisePSD   = dt**2/T*yw_max_all_freq;# scoresV;
        # signalPlusNoisePSDPulseGroup   = dt^2/T*scoresV;# scoresV;
        signalPSD             = signalPlusNoisePSD - ml.repmat(noisePSDAtZetas,1,n_pls);
        # signalPSDPulseGroup   = signalPlusNoisePSDPulseGroup-noisePSDAtZetas;
        
        signalPSD[signalPSD < 0] = 0; # Can't have negative values
        # signalPSDPulseGroup(signalPSD<0)   = 0; # Can't have negative values
        signalPowers           = signalPSD*fBinWidthZetas;
        signalAmps             = np.sqrt(signalPowers);

        np.seterr(divide = 'ignore') 
        SNRdB = 10 * np.log10(signalPSD / ml.repmat(noisePSDAtZetas,1,n_pls));
        np.seterr(divide = 'warn') 
        
        SNRdB[SNRdB == np.inf] = np.nan;
        
        # This block of code determines where there are peaks (negative
        # slopes to the left and right) and where there are valleys
        # (positive slopes to left and right)

        # greater_than_next = scoresV(1:end)>[scoresV(2:end);0];
        # greater_than_prev = scoresV(1:end)>[0;scoresV(1:end-1)];
        nextScores = np.concatenate((scoresV[1:],   [0]))
        prevScores = np.concatenate(([0],           scoresV[:-1]))
        greater_than_next = np.greater(scoresV, nextScores)
        greater_than_prev = np.greater(scoresV, prevScores)
        
        # slope_pos  = (~greater_than_next & greater_than_prev);
        # slope_neg  = ( greater_than_next &~greater_than_prev);
        # slope_val  = (~greater_than_next &~greater_than_prev);
        # slope_peak = ( greater_than_next & greater_than_prev);
        slope_pos  = np.logical_and(np.logical_not(greater_than_next),                greater_than_prev)
        slope_neg  = np.logical_and(               greater_than_next,  np.logical_not(greater_than_prev))
        slope_val  = np.logical_and(np.logical_not(greater_than_next), np.logical_not(greater_than_prev))
        slope_peak = np.logical_and(               greater_than_next,                 greater_than_prev)
        
        # score_left_bndry  = [true; ~isnan(scoresV(2:end)) & isnan(scoresV(1:end-1))];
        # score_right_bndry = [~isnan(scoresV(1:end-1)) & isnan(scoresV(2:end)); true];
        score_left_bndry  = np.concatenate(([True], np.logical_and(np.isfinite(scoresV[1:]), np.isnan(scoresV[:-1]))))
        score_right_bndry = np.concatenate((np.logical_and(np.isfinite(scoresV[:-1]), np.isnan(scoresV[1:])), [True]))
                    
        # [score_left_bndry&slope_neg,score_right_bndry&slope_pos,scoresV]
        # This deals with the frequency masking and some scoresV will be
        # NaN because they were excluded from consideration in the
        # incohsum processing. If the score is a left boundary
        # (frequencies lower were excluded) and the slope is negative,
        # this would be considered a peak. Visa-versa for right
        # boundaries. 
        # slope_peak = slope_peak|(score_left_bndry&slope_neg)|(score_right_bndry&slope_pos);
        slope_peak = np.logical_or(slope_peak, np.logical_or(np.logical_and(score_left_bndry, slope_neg), np.logical_and(score_right_bndry, slope_pos)))

        # How many time windows difference do we considered the found
        # pulse to be the same as one found at a different frequency?
        # We'll say that if they are within two pulse time width of 
        # each other they are the same pulse. 
        diff_thresh = 2*obj.n_p/obj.n_ws;
        p = 0;                 # Initilize a peak index variable
        curr_scores = scoresV;# Initially these are the same.
        
        # The peak and peak_ind variables are a vector of scoresV and the
        # frequency index of the identified peaks of the pulses. There
        # could only ever be n_freqs peaks, so we preallocate the
        # vectors to be that long, although there will typically be far
        # fever of them - ideally the number of tags in the signal. 
        peak              = np.zeros(n_freqs)
        peak_ind          = np.zeros(n_freqs, dtype=np.int_)
        bandwidth_of_peak = np.zeros(n_freqs)

        # msk               = false(n_freqs,n_freqs);
        # indiv_msk         = false(n_freqs,n_freqs);
        msk               = np.zeros((n_freqs, n_freqs)).astype(np.bool_)
        indiv_msk         = np.zeros((n_freqs, n_freqs)).astype(np.bool_)
        
        # peak_masked_curr_scores is a vector of scoresV where only the
        # peaks have scoresV. All others (valleys, +slope, -slope) are
        # set to zero. As we identify peaks, they will also be set to
        # zero so that we can work our way through these peaks.
        peak_masked_curr_scores = curr_scores * slope_peak;
        # This checks to make sure that at least one frequency has a
        # threshold exceeding score. If there isn't one (all the scroes
        # are below the threshold), we set the output variables as NaN
        # so that the caller knows nothing was detected.        
        
            # 2021-10-11
        # Update to to set all peak_masked_curr_scores less than the
        # threshold to zero. This wasn't needed when there was one
        # threshold, but with local frequency specific thresholds, we
        # only want threshold positive peaks listed in this vector so
        # the while loop below doesn't try to sort out sub-threshold
        # peaks. This can happen when there is a sub threhold peak that
        # has a higher score than a score at a different frequency that
        # was lower, but exceeded it's local threshold. 
        peak_masked_curr_scores = peak_masked_curr_scores * (peak_masked_curr_scores >= thresh)
                
        # if ~any(peak_masked_curr_scores>thresh, 'all')
        #     # peak_ind = [];
        #     peak_ind = NaN(1,1);
        #     peak     = NaN(1,1);# [];
        # end
        if not np.any(np.greater(peak_masked_curr_scores, thresh)):
            peak_ind = np.nan
            peak     = np.nan
        
        # Keep doing this loop below while there are scoresV that exceed
        # the threshold which aren't masked as a valley, +slope, -slope,
        # or previously identified peak/sideband. 

        # while any(peak_masked_curr_scores>thresh, 'all')
        while np.any(np.greater(peak_masked_curr_scores, thresh)):
            #    hold on; plot3([1:160],ones(160,1)*p,curr_scores)
            # Identify the highest scoring peak of the currently
            # identifed scoresV. 

            # [peak(p), peak_ind(p)] = max(peak_masked_curr_scores);
            maxIndex    = np.argmax(peak_masked_curr_scores)
            peak_ind[p] = maxIndex
            peak[p]     = peak_masked_curr_scores[maxIndex]

            # Build a mask that highlights all the elements whose time
            # windows share (or are close) to any of the time windows
            # of the pulses associated with the current peak.

            # for k= 1:n_blks
            for k in range(n_blks):
                # This block of code is used to find the number of time
                # windows between the current peak and the others in
                # the same block. It checks the blocks in front and
                # behind 
                
                # n_diff_to_curr_pk is a vector of time window
                # differences between the currently identified peaks and
                # all the other identified time windows in the current
                # block

                # n_diff_to_curr_pk = abs(S_cols(peak_ind(p),k)-S_cols(:,k));
                # n_diff_check_back = nan(n_freqs,1);
                # n_diff_check_for  = nan(n_freqs,1);
                n_diff_to_curr_pk = np.abs(S_cols[peak_ind[p], k-1] - S_cols[:,k-1])
                n_diff_to_curr_pk = np.abs(S_cols[peak_ind[p], k-1] - S_cols[:,k-1])
                n_diff_check_back = np.zeros(n_freqs)
                n_diff_check_back.fill(np.nan)
                n_diff_check_for  = np.zeros(n_freqs)
                n_diff_check_for.fill(np.nan)
                
                # Remember that the segments are long enough to
                # guarentee K pulses, but can get K+1. If we identified
                # the first or last K peaks, but there is an extra, its
                # side bands might be in this list in nearby frequencies.
                # We need to exclude these like we do the sidebands that
                # are nearby in time. In this case, we want to see if
                # any of the identified peaks are +/-tp+/-tip from the
                # current peak. Here we look a block forward and/or 
                # backward at the reported columns of the S matrix that 
                # were identified as exceeding the decision threshold 
                # and see if those columns are the same as the current
                # blocks identified column. We'll check later in the
                # code to see if these are less than a threshold.
 
                # if k<=n_blks-1 # Don't compute forward check when k=n_blks
                #     n_diff_check_for   =  abs(S_cols(peak_ind(p),k)-S_cols(:,k+1));
                # elseif k>=2 # Don't compute backward check when k=1
                #     n_diff_check_back  =  abs(S_cols(peak_ind(p),k)-S_cols(:,k-1));
                # end
                if k <= n_blks-1: # Don't compute forward check when k=n_blks
                    n_diff_check_for   =  np.absolute(S_cols[peak_ind[p], k-1] - S_cols[:, k]);
                elif k >= 2: # Don't compute backward check when k=1
                    n_diff_check_back  =  np.absolute(S_cols[peak_ind[p], k-1] - S_cols[:, k-2]);
                
                # There could be instances when a single block contains
                # two pulses, as mentioned above, but if the search is
                # only for one pulse (K = 1) then the if statments above
                # won't run. We need here to check if there are
                # identified scoresV within the current block that are
                # also one time repetition interval (tip+/tipu) away
                # from the current peak
                
                # diff_check_curr_blk_frwd = (S_cols(:,k) < S_cols(peak_ind(p),k) +N+M) & (S_cols(:,k) > S_cols(peak_ind(p),k) +N-M);
                # diff_check_curr_blk_bkwd = (S_cols(:,k) > S_cols(peak_ind(p),k) -N-M) & (S_cols(:,k) < S_cols(peak_ind(p),k) -N+M);
                # diff_check_curr = diff_check_curr_blk_frwd | diff_check_curr_blk_bkwd;
                a = np.less     (S_cols[:, k-1], S_cols[peak_ind[p], k-1] +N+M)
                b = np.greater  (S_cols[:, k-1], S_cols[peak_ind[p], k-1] +N-M)
                diff_check_curr_blk_frwd = np.logical_and(a, b)
                a = np.greater  (S_cols[:, k-1], S_cols[peak_ind[p], k-1] -N-M)
                b = np.less     (S_cols[:, k-1], S_cols[peak_ind[p], k-1] -N+M)
                diff_check_curr_blk_bkwd = np.logical_and(a, b)
                diff_check_curr = np.logical_or(diff_check_curr_blk_frwd, diff_check_curr_blk_bkwd)
                
                # This code below builds a mask that highlights the 
                # sidebands of the peak. It looks for the frequency bin
                # (higher and lower) where the score no longer meets 
                # the threshold. If no threshold crossings are found, 
                # it assumes the entire frequeny list is a sideband. 
                # This happens often in an informed search method when
                # the frequency search has been narrowed to a small 
                # band on either side of a known peak.
                
                # These are the number of frequency bins foward or
                # backward from the peak to the last score>threshold
                # inds_bkwd_2_thresh_xing = find(scoresV(peak_ind(p)-1:-1:1)<thresh,1,'first')-1;
                # inds_frwd_2_thresh_xing = find(scoresV(peak_ind(p)+1:end)<thresh,1,'first')-1;

                # inds_bkwd_2_thresh_xing = find(scoresV(peak_ind(p)-1:-1:1, 1) < thresh(peak_ind(p)-1:-1:1, 1), 1, 'first') -1;
                a = np.less(scoresV[np.arange(peak_ind[p]-1, 2, -1)], thresh[np.arange(peak_ind[p]-1, 2, -1)])
                inds_bkwd_2_thresh_xing = ml.findV(a, 1, 'first') - 1

                # inds_frwd_2_thresh_xing = find(scoresV(peak_ind(p)+1:end, 1) < thresh(peak_ind(p)+1:end, 1), 1, 'first') -1;
                a = np.less(scoresV[peak_ind[p]+1:], thresh[peak_ind[p]+1:])
                inds_frwd_2_thresh_xing = ml.findV(a, 1, 'first') - 1
                
                # Here we look for the location where the slope changes.
                # We don't use the -1 like above because we want to be
                # sure to capture the entire saddle between peaks, so
                # we include the point after the transition of the slope
                # sign. If we didn't do this, then the lowest point
                # between the peaks that was still above the threshold
                # wouldn't be included in a sideband mask and might get
                # picked up later as an additional peak. 

                # inds_bkwd_2_next_valley = find(slope_val(peak_ind(p)-1:-1:1,1)==1,1,'first');
                inds_bkwd_2_next_valley = ml.findV(slope_val[np.arange(peak_ind[p]-1, 2, -1)] == 1, 1, 'first');

                # inds_frwd_2_next_valley = find(slope_val(peak_ind(p)+1:end,1)>0,1,'first');
                inds_frwd_2_next_valley = ml.findV(slope_val[peak_ind[p]+1:] > 0, 1, 'first');
                
                # Wf_sub = Wf(freq_mask);
                
                # if isempty(inds_frwd_2_thresh_xing) && isempty(inds_bkwd_2_thresh_xing)
                if len(inds_frwd_2_thresh_xing) == 0 and len(inds_bkwd_2_thresh_xing) == 0:
                    # If it can't find a place backwards or forwards
                    # that is less than the threshold, the whole thing
                    # could be sideband. Often the case for informed
                    # search. 
#                         sideband_msk = true(n_freqs,1);
                    
                    # Alternatively, the interactions of
                    # adjacent sidebands might be so large, that there
                    # are no adjacent scoresV that are below the
                    # threshold. We need to deal with both of these
                    # cases. In the latter, we need to find the
                    # frequency bin ahead of and behind the peak where
                    # the slope changes and define that as the
                    # sidebands. If the slope doesn't change, then we
                    # are likely the former case wherein the entire set
                    # of scoresV are side band and we make the mask all
                    # true. 

                    # sideband_msk = false(n_freqs,1);
                    # if isempty(inds_bkwd_2_next_valley) && isempty(inds_frwd_2_next_valley)
                    #     sideband_msk = true(n_freqs,1);
                    # elseif isempty(inds_bkwd_2_next_valley) && ~isempty(inds_frwd_2_next_valley)
                    #     sideband_msk(1:peak_ind(p)) = true;
                    # elseif ~isempty(inds_bkwd_2_next_valley) && isempty(inds_frwd_2_next_valley)
                    #     sideband_msk(peak_ind(p):end) = true;
                    # elseif ~isempty(inds_bkwd_2_next_valley) && ~isempty(inds_frwd_2_next_valley)
                    #     sideband_msk(peak_ind(p)+...
                    #         (-inds_bkwd_2_next_valley:...
                    #         inds_frwd_2_next_valley))...
                    #         = true;
                    # end
                    sideband_msk = np.zeros(n_freqs).astype(np.bool_)
                    if len(inds_bkwd_2_next_valley) == 0 and len(inds_frwd_2_next_valley) == 0:
                        sideband_msk = p.ones(n_freqs).astype(np.bool_)
                    elif len(inds_bkwd_2_next_valley) == 0 and len(inds_frwd_2_next_valley) != 0:
                        sideband_msk[0 : peak_ind[p]] = True
                    elif len(inds_bkwd_2_next_valley) != 0 and len(inds_frwd_2_next_valley) == 0:
                        sideband_msk[peak_ind[p] : -1] = True
                    elif len(inds_bkwd_2_next_valley) != 0 and len(inds_frwd_2_next_valley) != 0:
                        sideband_msk[peak_ind[p] + ml.siVector(-inds_bkwd_2_next_valley, inds_frwd_2_next_valley)] = True
                    
                # If there was a threhold crossing before and/or after 
                # the peak, sort out the sidebands here
                else:
                    sideband_msk = np.zeros(n_freqs).astype(np.bool_)

                    # if isempty(inds_frwd_2_thresh_xing) && ~isempty(inds_bkwd_2_thresh_xing)
                    if len(inds_frwd_2_thresh_xing) == 0 and len(inds_bkwd_2_thresh_xing) != 0:
                        # Threshold crossing back was found but not
                        # forwards.
                        # What is smaller, the distance from the peak to
                        # the first entry backwards, or the distance
                        # from the peak to the threshold crossing in
                        # front of the peak? Use that distance as the
                        # 1/2 width of the sideband.

                        # result = min([inds_bkwd_2_next_valley,inds_bkwd_2_thresh_xing],[],'all');
                        # result = min([inds_bkwd_2_next_valley,inds_bkwd_2_thresh_xing],[],'all');
                        result = np.min([inds_bkwd_2_next_valley, inds_bkwd_2_thresh_xing]);
                        result = np.min([inds_bkwd_2_next_valley, inds_bkwd_2_thresh_xing]);

                        bandwidth_of_peak[p] = result
                    # elseif ~isempty(inds_frwd_2_thresh_xing) && isempty(inds_bkwd_2_thresh_xing)
                    elif len(inds_frwd_2_thresh_xing) != 0 and len(inds_bkwd_2_thresh_xing) == 0:
                        # Threshold crossing forward was found but not
                        # backwards.
                        # What is smaller, the distance from the peak to
                        # the last entry forwards, or the distance
                        # from the peak to the threshold crossing in
                        # back of the peak? Use that distance as the
                        # 1/2 width of the sideband.

                        # result = min([inds_frwd_2_next_valley,inds_frwd_2_thresh_xing],[],'all');
                        result = np.min([inds_frwd_2_next_valley, inds_frwd_2_thresh_xing])

                        bandwidth_of_peak[p] = result
                    else:
                        # Threshold crossing forward and backward was
                        # found.
                        # What is smaller, the distance
                        # from the peak to the threshold crossing in
                        # or the distance from the peak to the nearby 
                        # valley? Use that distance as the 1/2 width of
                        # the sideband. Do this both forwards and
                        # backwards. 

                        # if ~isempty(inds_bkwd_2_next_valley)
                        #     bkwd_BW = min([inds_bkwd_2_thresh_xing,inds_bkwd_2_next_valley],[],'all');
                        # else
                        #     bkwd_BW = inds_bkwd_2_thresh_xing;
                        # end
                        # if ~isempty(inds_frwd_2_next_valley)
                        #     frwd_BW = min([inds_frwd_2_next_valley,inds_frwd_2_thresh_xing],[],'all');
                        # else
                        #     frwd_BW = inds_frwd_2_thresh_xing;
                        # end
                        if len(inds_bkwd_2_next_valley) != 0:
                            bkwd_BW = np.min([inds_bkwd_2_thresh_xing, inds_bkwd_2_next_valley])
                        else:
                            bkwd_BW = inds_bkwd_2_thresh_xing
                        if len(inds_frwd_2_next_valley) != 0:
                            frwd_BW = np.min([inds_frwd_2_next_valley, inds_frwd_2_thresh_xing])
                        else:
                            frwd_BW = inds_frwd_2_thresh_xing
                        
                        # bandwidth_of_peak(p) = max([bkwd_BW,frwd_BW],[],'all');
                        bandwidth_of_peak[p] = np.max([bkwd_BW, frwd_BW])

                    # Make sure we aren't requesting masking of elements
                    # that are outside the bounds of what we have 
                    # available in the vector

                    # lower_sideband_ind = max([1,peak_ind(p)-bandwidth_of_peak(p)],[],'all');
                    # upper_sideband_ind = min([n_freqs,peak_ind(p)+bandwidth_of_peak(p)],[],'all');
                    # sideband_msk(lower_sideband_ind:upper_sideband_ind) = true;
                    lower_sideband_ind = np.max([0, peak_ind[p] - bandwidth_of_peak[p]])
                    upper_sideband_ind = np.min([n_freqs-1, peak_ind[p] + bandwidth_of_peak[p]])
                    sideband_msk[ml.siVector(lower_sideband_ind, upper_sideband_ind)] = True
                
                # Here we build up a mask that incorporates all the
                # peaks and their sidebands that we have identified 
                # so far. The msk(:,p+1) entry is here because we are
                # looping through n_blks and are updating the p+1 column
                # each time. 
                
                # msk(:,p+1) = msk(:,p) | msk(:,p+1) | \
                #               (sideband_msk | \
                #                   ((n_diff_to_curr_pk<=diff_thresh)|(n_diff_check_back<=diff_thresh)|(n_diff_check_for<=diff_thresh))|diff_check_curr);                 
                a = msk[:, p]
                b = msk[:, p+1]
                firstOr = np.logical_or(a, b)
                c = n_diff_to_curr_pk <= diff_thresh
                d = n_diff_check_back <= diff_thresh
                e = n_diff_check_for  <= diff_thresh
                secondOr = np.logical_or(np.logical_or(np.logical_or(c, d), e), diff_check_curr)
                thirdOr = np.logical_or(sideband_msk, secondOr)
                msk[:,p+1] = np.logical_or(firstOr, thirdOr)

            # Extract the mask for this peak and no others

            # indiv_msk(:,p) = msk(:,p+1)-msk(:,p);
            indiv_msk[:,p] = msk[:,p+1] ^ msk[:,p]

            # Update the current scoresV to exclude peaks and sidebands
            # we have identified so far.

            # curr_scores = scoresV.*~msk(:,p+1); # Mask the recently found sidebands                
            # peak_masked_curr_scores = curr_scores.*slope_peak; # Mask to only look at peaks
            # peak_masked_curr_scores = peak_masked_curr_scores.*(peak_masked_curr_scores>=thresh);# Eliminate non-threshold exceeding scoresV from consideration
            curr_scores = scoresV * np.logical_not(msk[:,p+1]); # Mask the recently found sidebands                
            peak_masked_curr_scores = curr_scores * slope_peak; # Mask to only look at peaks
            peak_masked_curr_scores = peak_masked_curr_scores * np.greater_equal(peak_masked_curr_scores, thresh); # Eliminate non-threshold exceeding scoresV from consideration

            p = p + 1
            if p >= msk.shape[1] - 1:
                break
        
        # Clean up the msk and indiv_mask so their columns align with
        # n_pulsegroup_found. 

        # n_pulsegroups_found = p-1;
        # msk       = circshift(msk,-1,2);
        # msk       = msk(:,1:n_pulsegroups_found);
        # indiv_msk = indiv_msk(:,1:n_pulsegroups_found);
        n_pulsegroups_found = p
        msk                 = ml.circshift(msk, -1, 2)
        msk                 = msk[:, ml.siVector(0, n_pulsegroups_found)]
        indiv_msk           = indiv_msk[:, ml.siVector(1, n_pulsegroups_found)]

        if n_pulsegroups_found > 0:
            # Only update from NaN if there were some found. 
            # peak_ind  = peak_ind(1:n_pulsegroups_found);
            peak_ind  = peak_ind[ml.siVector(0, n_pulsegroups_found-1)]

        # Each row of msk is a mask that isolates the elements of the
        # yw_max_all_freq vector associated with that column's peak and all
        # higher power peaks. For exampl msk(:,2) provides a mask that
        # highlights which elements of yw_max_all_freq are associated in
        # time with the second highest peak and the first highest peak.
        # Plotting scoresV.*~msk(:,2) will plot all of the scoresV of
        # yw_max_all_freq that are not associated in time with the first
        # or second highest power peak.
        
        
        # #  CODE FOR REPORTING A CANDIDATE PULSE AT ALL THE FREQUENCY
        # # BINS WITHOUTH DOING THE THRESHOLDING.
        
        # Create a frequency array that accounts for the masking that
        # was done to reduce the frequency space.
        Wf_sub = obj.Wf[freq_mask];
        freq_found = obj.Wf;
        
        # t_found here is the start of the pulse - not the center like
        # in the time stamps in the stft, which are the centers of the
        # windows. The dt_stft/2 term offfset would affect both the first
        # and second terms of the equation below and therefore cancel.

        # t_found    = NaN(n_freqs,n_pls);
        # t_found(freq_mask,:)    = obj.stft.t(S_cols(freq_mask,:))-obj.stft.t(1)+obj.t_0;# Don't forget the add the t_0 of this waveform
        # f_bands    = [obj.Wf-(obj.Wf(2)-obj.Wf(1))/2, obj.Wf+(obj.Wf(2)-obj.Wf(1))/2];
        t_found                 = np.empty((n_freqs, n_pls))
        t_found.fill(np.nan)
        t_found[freq_mask, :]   = obj.stft.t[S_cols[freq_mask,:]] - obj.stft.t[0] + obj.t_0
        f_bandsStart            = obj.Wf - ((obj.Wf[1] - obj.Wf[0]) / 2)
        f_bandsEnd              = obj.Wf + ((obj.Wf[1] - obj.Wf[0]) / 2)
        
        # Preallocate cur_pl matrix
        cur_pl = np.empty((n_freqs, n_pls), dtype=object)

        # Build out the pulse object for each one found

        # for i = 1:n_pls
        #     for j = 1:n_freqs
        #         cur_pl(j,i) = makepulsestruc(signalAmps(j,i),...# NaN,...
        #             yw_max_all_freq(j,i),...
        #             SNRdB(j,i),...
        #             t_found(j,i),...
        #             t_found(j,i)+obj.ps_pre.t_p,...
        #             [t_found(j,i)+obj.ps_pos.t_ip-obj.ps_pos.t_ipu--obj.ps_pos.t_ipj,...
        #             t_found(j,i)+obj.ps_pos.t_ip+obj.ps_pos.t_ipu+obj.ps_pos.t_ipj],...
        #             freq_found(j),...
        #             f_bands(j,1),...
        #             f_bands(j,2));
        #         cur_pl(j,i).con_dec = false;    # Confirmation status
        #         if scoresV(j)>=thresh# scoresV(j)>=thresh(j)
        #             cur_pl(j,i).det_dec = true;
        #         end
        #     end
        # end
        for i in range(n_pls):
            for j in range(n_freqs):
                cur_pl[j,i] = Pulse( \
                                amplitude       = signalAmps        [j, i], \
                                yw              = yw_max_all_freq   [j, i], \
                                snr             = SNRdB             [j, i], \
                                startTime       = t_found           [j, i], \
                                endTime         = t_found           [j, i] + obj.ps_pre.t_p, \
                                nextTimeRange   = [t_found[j,i] + obj.ps_pos.t_ip - obj.ps_pos.t_ipu - -obj.ps_pos.t_ipj, \
                                                    t_found[j,i] + obj.ps_pos.t_ip + obj.ps_pos.t_ipu + obj.ps_pos.t_ipj], \
                                peakFreq        = freq_found        [j], \
                                freqStart       = f_bandsStart      [j], \
                                freqEnd         = f_bandsEnd        [j])
                cur_pl[j, i].con_dec = False;    # Confirmation status
                if np.any(thresh < scoresV[j]):
                    cur_pl[j,i].det_dec = True
        
        pl_out = cur_pl

        return [ pl_out, indiv_msk, peak_ind ]
