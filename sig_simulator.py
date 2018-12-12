from scipy import signal
import numpy as np
from numpy import random
import mpmath

def simu_sinusoid_sig(sps, x_len, amp=1, phase=0.0):
    x_range = [m/sps for m in range(0, x_len)]
    y = [mpmath.sin(x+phase) for x in x_range]
    return (x_range, y)

def simu_gaussian(x, mean=0, std=1):
    return np.random.normal(mean, std, len(x))


