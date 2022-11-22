#classdef pulsestats < handle
class PulseStats:
#    PULSESTATS Object contain parameter for a set of pulses. This class can be
#    used as a set of priori information that details the expectation of these
#    properties or built posteriori after detection. Some properties associated
#    with detections (pl, clst, cmsk,cpki) are not typically populated for
#    priori pulsestats object, unless they are known. The pulses listed in the
#    'pl' property depend on the selection mode selected in waveform process
#    method, but the items in clst, cmsk, and cpki, properties here always
#    contain all of the candidate pulses, mask, and peak indicies for the
#    waveform that was processed. The details of how these are specified can be
#    see in the process method of the waveform class.
#    
#    PROPERTIES:
#       t_p     Duration of pulse (second)
#               signal energy contained for 0<=t<tp
#       t_ip    Inter-pulse time (seconds)
#       t_ipu   Inter-pulse time uncertainty(seconds)
#       t_ipj   Inter-pulse time jitter (deviations from means) (seconds)
#       fp      Pulses' peak frequency (Hz) (highest power)
#       fstart  Pulses' lower frequency bound
#       fend    Pulses' upper frequency bound
#       tmplt   Time domain amplitude template of the baseband pulse
#               described with any number of evenly space time points
#               between 0 and tp. [1 1] would represent a rectangular
#               pulse of duration tp. [1 1 -1 -1] would be a constant
#               1 from 0-tp/3, a ramp from tp/3-2tp/3, and a constant
#               -1 from 2tp/3-tp.
#       mode    What tracking mode should be use (for priori) or was used
#               (for posteriori). 'D' 'C' or 'T'
#       pl      'Pulse List' Vector of pulses objects in waveform (if
#               known). This is the pulse group currently being
#               tracked by the algorithm. The items in this list depend on the
#               'selection_mode' used in the in the waveform process method.
#       clst    'Candidate List' is a matrix of pulses objects in the
#               waveform (if known). Pulses objects in row i are those
#               found at ith frequeny of the Wf frequency vector.
#               Pulses objects in column j are those found at jth time
#               window of the STFT for the processed segment. This has
#               as many columns as K (number of pulses) for the
#               waveform segment. Has as many rows as frequencies used
#               in the processing. Most of these 'pulses' will not
#               meet the decision threshold and are just noise.
#               Others might be sidebands. Some will be true pulses.
#        cmsk   'Candidate List Mask' is a matrix the contains masks for
#               each of the pulses that met the threshold. These masks
#               are true for frequencies that the peeling algorithm
#               deteremined to be sidebands of the true peak. These
#               were determined through time correlations with the
#               peaks. This matrix can be used to exclude candidate
#               pulses in clst from further consideration. There is one
#               row in cmsk for each frequency (like clst), but the
#               number of columns corresponds to the number of peaks
#               found in the processing result that met the decision
#               criteria. For example, cmsk might have true values in
#               column 2 at rows 5-8. This would indicated that
#               the candidate pulses in clst in rows 5-8 correspond to
#               the peak and sidebands of the #2 pulse listed cpki.
#         cpki  These are the row indices of clst that were found to be
#               peak (the center of the pulse frequency). Sidebands
#               accociated with these peaks are recorded in each
#               column cmsk. For example, the if the second element of
#               cpki is 10, then the 10th row (frequency index) of clst
#               contains a set of pulses that meet the decision
#               criteria. The sidebands of this pulse group would be
#               found in the 2nd column of cmsk.
#       thresh  A vector of the thresholds computed for this used as the
#               decision criteria for each of the candidate pulses in clst
    
#    properties
#        t_p   (1, 1) double %Duration of pulse (second)
#        t_ip  (1, 1) double %Inter-pulse time (seconds)
#        t_ipu (1, 1) double %Inter-pulse time uncertainty (seconds)
#        t_ipj (1, 1) double %Inter-pulse time jitter (deviations from means) (seconds)
#        fp    (1, 1) double %Pulses' peak frequency (Hz) (highest power)
#        fstart(1, 1) double %Pulses' lower frequency bound
#        fend  (1, 1) double %Pulses' upper frequency bound
#        tmplt (1, :) double %Time domain amplitude template of the baseband pulse
#        mode  (1, 1) char   %The tracking mode that should be use (for priori) or was used (for posteriori). 'D' 'C' or 'T'
#        pl    (1, :)        %'Pulse List' Vector of pulses objects in waveform
#        clst  (:, :)        %'Candidate List' is a matrix of pulses objects in the waveform
#        cmsk  (:, :) logical%'Candidate List Mask' is a matrix the contains masks for each of the pulses that met the threshold.
#        cpki  (:,:)  double %These are the row indices of clst that were found to be peak (the center of the pulse frequency).
#    end

    def __init__ (self, t_p = 0, t_ip = 0, t_ipu = 0, t_ipj = 0, fp = 0, tmplt = [ 1, 1 ]):
        self.t_p    = t_p
        self.t_ip   = t_ip
        self.t_ipu  = t_ipu
        self.t_ipj  = t_ipj
        self.fp     = fp
        self.fstart = None
        self.fend   = None
        self.tmplt  = tmplt
        self.mode   = 'D'
        self.pl     = None
        self.clst   = None
        self.cmsk   = None
        self.cpki   = None
        if self.t_p == 0:
            print("PulseStats() self:%x" % id(self))
            raise

    def __str__(self): 
        return "PulseStats(%x): t_p:%f t_ip:%f t_ipu:%f t_ipj:%f fp:%f" % (id(self), self.t_p, self.t_ip, self.t_ipu, self.t_ipj, self.fp)
