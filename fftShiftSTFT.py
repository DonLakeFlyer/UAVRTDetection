import matlab

import numpy as np
import math
import scipy

# function [xout] = fftshiftstft(x)
def fftShiftSTFT(x: np.ndarray):
    # FFTSHIFTSTFT Performs a standard fftshift operation but accounts for the
    # different frequency range definition in Matlabs STFT function when
    # outputing a 'centered' frequency range
    #    When calling the STFT function, the default output for frequency is
    #    the 'centered' method where frequencies are reported on the range of
    #    (-n/2+1:n/2)*Fs/n if the vector is of even length and
    #    (ceil(-n/2):floor(n/2))*Fs/n if the vector is of odd length. It seems
    #    that Matlab's fftshift function is develped for frequencies in the
    #    range of (-n/2:n/2+1)*Fs/n if the vector is of even length. The odd
    #    length definition is unaffected. The result is an incorrect shifting of
    #    the frequency vector if fftshift is used on f defined by stft's
    #    centered method. This function uses the fftshift command for vectors
    #    of even length, but shifts the result by 1 to the left to correct for
    #    this problem. 
    #    
    #    If x is a matrix, it is assumed that the matrix is formatted similar to
    #    the output of a STFT call, wherein the rows represent the different
    #    frequencies and the rows are the time windows. As such, in this case
    #    the rows are operated on (shifted). 
    # 
    # INPUTS:
    #    x       nxm     A matrix of real or complex data. If passed a matrix,
    #                    the rows should represent frequency bins
    # OUTPUT:
    #    xout    nxm     A matrix of real or complex shifted data
    # # 
    #  Author: Michael Shafer
    #  Date:   2020-12-19
    # 
    # #  Change log:
    #  2022-01-10    Updated to deal with row or column vectors or stft matrices
    # 
    # # 

    if x.ndim > 2:
        raise Exception("Only accepts 1D or 2D data")

    # sz = size(x);
    # if sz(1)==1 && sz(2)~= sz(1) # Passed a row vector
    #     dim2shift = 2;
    # else # Passed a matrix or a column vector
    #     dim2shift = 1;
    # end
    # n = numel(x);
    # if mod(n,2)
    #     xout = fftshift(x,dim2shift);    
    # else
    #     xout = circshift(fftshift(x,dim2shift),-1,dim2shift);
    # end

    n = matlab.numel(x)
    noCircshift = not math.fmod(n, 2)

    if x.ndim == 1:
        # row vector
        if noCircshift:
            return scipy.fft.fftshift(x)
        else:
            xout = np.roll(scipy.fft.fftshift(x), 1)
    else: 
        # matrix
        if noCircshift:
            return scipy.fft.fftshift(x, axes = (1,))
        else:
            xout = np.roll(scipy.fft.fftshift(x, axes = (1,)), 1, axis = 1)
