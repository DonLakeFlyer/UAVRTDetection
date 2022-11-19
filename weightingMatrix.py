from  makeSTFTFreqVector import *
import matlab
from fftShiftSTFT import *
from ifftShiftSTFT import *

import numpy as np
import scipy as sp
import math

# function [W,Wf] = weightingmatrix(x_of_n_in,Fs,zetas,frequencyRangeType)
def weightingMatrix(x_of_n_in: np.ndarray, Fs: float, zetas: np.ndarray, frequencyRangeType:str):
    # WEIGHTINGMATRIX Builds the spectral weighting matrix for the incoherent
    # summationation algorithm. The matrix is based on the DFT coefficients
    # developed from the waveform passed to the function. The function first
    # developeds a weighting vectors based on an unaltered waveform, but then
    # shifts the waveforms spectral content base fractions of the DFT bin width,
    # as defined by the zetas input. These shifted waveforms then have their
    # DFT's computed. The original and shifted DFT coefficients are then aligned
    # in a matrix. In this way, this weighting matrix allows for the dominant
    # frequency in the vector it weights to be between bins. This submatrix is
    # then row shifted and concantinated a number of times so that the DFT
    # coefficients are shifted across all DFT frequency bins. The outputs are 
    # the weighting matrix W and a frequency vector Wf that corresponds to the 
    # rows of W. The frequencies, and therefore the rows of the W matrix, use a
    # centered frequency definitions defined by the STFT.m algorithm. 
    # 
    #    Example:   [W,Wf] = weightingmatrix(rand(1,10),1,[0 0.5])
    #  
    #    INPUTS:
    #    x_of_n      nx1     Vector of discrete data sampled as Fs. Can be 1xn.
    #    Fs          1x1     Sampling rate in Hz of the dataset
    #    zetas       mx1     A vector of fractions of the DFT frequency bin
    #                        width for which to compute and include in the
    #                        spectral weighting matrix output. If no interbin
    #                        shifting is desired, enter 0. Range of inputs
    #                        [0,1).
    #    frequencyRangeType  String of either 'centered' or 'twosided'. See
    #                        stft.m documentation on details of these frequency
    #                        bounds. If 'centered' is selected, the zero
    #                        frequency point is near the center of the output
    #                        matrix. If 'twosided' is selected, the zero
    #                        frequency points is in the upper left corner of the
    #                        output W matrix. 
    # 
    #    OUTPUTS:
    #    W           nxnm    A matrix of spectral weighting vectors along each
    #                        column. The hermitian of this matrix is designed to
    #                        be multiplied by a vector of DFT coefficients. The 
    #                        weights held in column p can be thought of as the
    #                        matched DFT coefficients if the signal were to have
    #                        been shifted by the template by Wf(p). 
    #    Wf          nmx1    A matrix of the frequency shifts corresponding to
    #                        each column of W 
    # 
    # 
    #  The intended use of the W matrix is to multiply it's hermetian by a FFT 
    #  vector, or STFT matrix with frequencies in the rows. 
    #           
    # 
    #    [W^h]*S         or          [W^h]*FFT
    # 
    #  The outputs of this multiplication will have n*m rows. The rows of the
    #  result there for are the "scores" of at frequency associated with that
    #  row. The row frequencies are dependent on the zetas. Consider an input
    #  series that had frequencies [-100 0 100 200]. If the zeta input was 
    #  [0 0.5], weightings for full and half frequencies steps would be 
    #  developed. 
    # 
    #        output freq     input freq
    #                            -100    0   100     200
    #                -150    [                           ]
    #                -100    [                           ]
    #                -50     [                           ]
    #                0       [            W^h            ]
    #                50      [                           ]
    #                100     [                           ]
    #                150     [                           ]
    #                200     [                           ]
    # 
    #    The first row of the matrix above would contain the weightings for the
    #    -100 0 100 and 200 Hz frequencie bins if the FFT (or STFT time window)
    #    had a signal at -150 Hz. Similarly, the second row would contain the 
    #    weightings for the -100 0 100 and 200 Hz frequencie bins if the FFT 
    #    (or STFT time window) had a signal at -100 Hz. 
    # 
    # 
    # 
    # 
    # Author: Michael Shafer
    # Date:   2020-12-19
    # 
    # #  Change Log
    #  2022-01-12    Updated to include twosided frequency range for output and
    #                updated the code associated with generating the unsorted W
    #                matrix to increase speed. Added use of makestftFreqVector
    #                function. 
    # 
    # 

    # Check the inputs
    if x_of_n_in.ndim != 1:
        raise Exception('Input x_of_n_in must be a vector')
    if Fs <= 0:
        raise Exception('Fs input must be a positive scalar. ')
    if max(zetas) >= 1:
        raise Exception('Maximum input interbin fractions (zeta) must be less than 1.')
    if min(zetas) < 0:
        raise Exception('Minimum input interbin fractions (zeta) must be greater than or equal to 0.')
    if frequencyRangeType != 'twosided' and frequencyRangeType != 'centered':
            raise Exception('UAV-RT: Frequency range type must be either twosided or centered')

    x_of_n = x_of_n_in

    nw = len(x_of_n);
    numZetas = len(zetas);
    # n = transpose(0:(nw-1));
    n = matlab.matlabRange(0, (nw - 1));

    # Develop a vector of frequencies that would result from the centered method
    # of the STFT function if it were computed on the x input. This is useful
    # for debugging the frequency outputs.
    # These are designed to develop the same freqs as the STFT centered method 
    # Check the "'FrequencyRange' — STFT frequency range" in the STFT help file
    # if mod(nw,2)==0
    #     freqs_orig = ((-nw/2+1):nw/2)*Fs/nw;
    # else
    #     freqs_orig = (ceil(-nw/2):floor(nw/2))*Fs/nw;
    # end

    # s	= 1i*zeros(nw,numZetas);
    # Xs  = 1i*zeros(nw,numZetas);
    s	= np.zeros((nw, numZetas), dtype = np.cdouble);
    Xs	= np.zeros((nw, numZetas), dtype = np.cdouble);

    # for i = 1:numZetas
    #     s(:,i)  = exp(1i*2*pi*zetas(i)*n./nw);
    #     currDFT = fft(s(:,i).*x_of_n);
    #     Xs(:,i) = fftshiftstft(currDFT / norm(currDFT));
    for i in range(numZetas):
        # We are filling in columns of data
        s[:, i]     = np.exp(2 * np.pi * zetas[i] * n / nw, dtype = np.cdouble);
        currDFT     = sp.fft.fft(s[:, i] * x_of_n);
        Xs[:, i]    = fftShiftSTFT(currDFT / matlab.norm(currDFT));

    # stackedToeplitzMatrices = complex(zeros(numZetas*nw,nw));
    # for i = 1:numZetas
    #     rowStart = (i-1)*nw+1;
    #     rowEnd = rowStart+nw-1;
    #     stackedToeplitzMatrices(rowStart:rowEnd,:) = toeplitz(Xs(:,i),[Xs(1,i); flipud(Xs(2:end,i))]);    
    # end
    stackedToeplitzMatrices = np.empty((numZetas * nw, nw), dtype = np.cdouble);
    np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    for i in range(numZetas):
        rowStart    = i * nw
        rowEnd      = rowStart + nw
        c           = Xs[:, i]
        r           = np.array(Xs[0, i])
        r           = np.append(r, np.flipud(Xs[1:, i]))
        stackedToeplitzMatrices[rowStart:rowEnd,:] = sp.linalg.toeplitz(c, r)

    # Reshape in this way will interweave rows of each sub matrix of the stack
    W = matlab.reshape(stackedToeplitzMatrices, nw, numZetas * nw);

    # Build a vector of the frequencies
    freqs = makeSTFTFreqVector(2 * nw, 'twosided', Fs);

    # Simulink (coder?) didn't like the new round inputs
    # freqs = round(freqs*1000)/1000;
    freqs = np.round(freqs * 1000) / 1000;

    # Shift everything so we use a negative frequencies

    # nyq_ind  = find(freqs == floor(Fs/2), 1, 'first');
    # nyq_ind  = find(freqs == floor(Fs/2), 1, 'first');
    nyq_ind  = matlab.find(freqs == math.floor(Fs/2), 1, 'first');
    nyq_ind  = matlab.find(freqs == math.floor(Fs/2), 1, 'first');

    # frequency_shift_amount = -freqs(end)-(Fs-freqs(end));
    frequency_shift_amount = -freqs[-1]-(Fs-freqs[-1]);

    # zero_padding = zeros(1,nyq_ind);
    # wrapper = [zero_padding,frequency_shift_amount*ones(1,numel(freqs)-nyq_ind)];
    # Wf = wrapper' + freqs';
    Wf = np.zeros(nyq_ind)
    np.append(Wf, frequency_shift_amount * np.ones(len(freqs) - nyq_ind))
    np.append(Wf, freqs)
    Wf.transpose()

    # Here we sort the output to set up to have an ascending order of frequency
    # to be similar to the centered frequency list used elsewhere in the
    # codebase for the STFT output. 

    [ Wf,inds ] = matlab.sort(Wf);
    W = W[:, inds]

    if frequencyRangeType == 'twosided':
        W = ifftShiftSTFT(W);                       # Shift the rows to twosided from centered
        W = np.transpose(ifftShiftSTFT(np.transpose(W))); # Shift the colums to two sided from centered

    return [W, Wf]