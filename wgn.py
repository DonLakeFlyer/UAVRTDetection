import numpy as np

def wgn(m, n, power):
    # equates to matlab wgn(m, n, power, 'linear', 'complex')
    # FIXME: Hack to get things running. Need real implementation
    np.ones((m, n), dtype=np.cdouble)
