import numpy as np
from scipy.optimize import fsolve

# Comes from here: https://github.com/SeanPluemer/Matlab_to_python/blob/main/evfit.py

def evfit(x, alpha = 0.05, censoring=0, freq=0, options=0):
    #
    '''This function is translated from Matlab
    EVFIT Parameter estimates and confidence intervals for extreme value data.
    PARMHAT = EVFIT(X) returns maximum likelihood estimates of the
    parameters of the type 1 extreme value distribution given the data in X.
    PARMHAT(1) is the location parameter, mu, and PARMHAT(2) is the scale parameter, sigma.
    '''


    if not isinstance(x, list):
        print("stats:evfit:VectorRequired")
        return 0
    if censoring==0:
        censoring = np.zeros(len(x) )
    else: #todo, if censoring not empty and len(x) and len(censoring not same size throw error)
        print('stats:evfit:XCensSizeMismatch')
        return 0
    if freq==0:
        freq = np.ones(len(x))
    else:
        pass
    '''% The default options include turning fzero's display off.  This function
     gives its own warning/error messages, and the caller can turn display on
     to get the text output from fzero if desired.
    '''
    if options==0:
        pass
    else:
        pass
    classX = type(x[0])

    n = sum(freq)
    temp = freq.dot(censoring)
    ncensored = temp
    if not temp==0:
        ncensored =  sum(freq.dot(censoring))
    nuncensored = n - ncensored
    #get range between high and low
    rangex = max(x) - min(x)
    maxx = max(x)

    if (n==0) and (nuncensored==0) and (np.isfinite(rangex)):
        pass
    elif ncensored==0:
        #The likelihood surface for constant data has its maximum at the
        #boundary sigma==0.  Return something reasonable anyway.
        if rangex < 2.2251e-308:
            #this checks to see if rangex is below min double value
            pass
        #data is oka
    else:
        # When all uncensored observations are equal and greater than all the
        #censored observations, the likelihood surface has its maximum at the
        #boundary sigma==0.  Return something reasonable anyway.
        pass
    #Shift x to max(x) == 0, min(x) = -1 to make likelihood eqn more stable.
    test = [i - maxx for i in x]
    x0 = np.array(test) / np.array(rangex)
    #% First, get a rough estimate for the scale parameter sigma as a starting value.
    # Use MM when there is no censoring
    if ncensored==0:
        sigmahat = (np.sqrt(6)*np.std(x0))/np.pi
        wgtmeanUnc = sum(np.array(freq)*np.array((x0))) / n
    # Bracket the root of the scale parameter likelihood eqn ...
    if (lkeqn(sigmahat, x0, freq, wgtmeanUnc) > 0):
        upper = sigmahat
        lower = 0.5 * upper
        while (lkeqn(lower, x0, freq, wgtmeanUnc)>0):
            upper = lower
            lower = 0.5 * upper
            #if lower <x

    else:
        lower = sigmahat
        upper = 2*lower
        while (lkeqn(upper,x0,freq,wgtmeanUnc) <0):
            lower = upper
            upper = 2 * lower
            if upper > max(classX): #overflow
                print('stats:evfit:NoSoluton')
                return 0

    bnds = [lower, upper]
    sigmahat = fsolve(lkeqn,bnds[0], args=( x0, freq, wgtmeanUnc))
    x = freq.dot(np.exp(np.array(x0) / sigmahat))

    muhat = sigmahat.dot(np.log(x / nuncensored));
    parmhat =  []
    parmhat.append((rangex * muhat) + maxx)
    parmhat.append(rangex * sigmahat)

    return parmhat


def lkeqn(sigma, x, w, xbarWgtUnc):
    w = np.array(w) * np.array(np.exp(np.array(x)/np.array(sigma)))
    v = np.array(sigma) + np.array(xbarWgtUnc) - sum(np.array(x)*np.array(w)) / sum(np.array(w))

    return v