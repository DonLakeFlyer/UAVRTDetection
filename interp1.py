from scipy.interpolate import interp1d

def interp1(x, v, xq, kind = 'linear', fill_value = None):
    func = interp1d(x, v, kind = kind, fill_value = fill_value)
    return func(xq)