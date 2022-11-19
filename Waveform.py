import PulseStats
import Threshold
import WaveformSTFT
import matlab
from getTemplate import getTemplate
from weightingMatrix import weightingMatrix

import numpy as np
import math

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
        obj.ps_pos  = ps_pre.copy() # Current stats are same as previous during construction

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
        print("Waveform.stft.S.shape", obj.stft.S.shape)

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
        ind_start  = matlab.find(t <= t0, 1, 'last');
        
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
                wind_start          = matlab.find(obj.stft.t <= t0_wind, 1, 'last');
                winds4cleavedwfm    = wind_start - 1 + np.arange(windows_in_cleaved_wfm);

                wfmout.stft         = obj.stft.copy();     # Set to entire object to get all props
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
