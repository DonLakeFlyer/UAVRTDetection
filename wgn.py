import numpy as np

def wgn(m, n, power):
    # FIXME: Hack to get things running. Need real implementation
    rng = np.random.default_rng()
    return np.random.normal(0, 1, size = (n, m)).astype(np.cdouble)
    rng.normal(scale=np.sqrt(noise_power), size=time.shape)
