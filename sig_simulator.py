from scipy import signal
import numpy as np
from numpy import random
import math


def simu_gaussian(mean=0, std=1, shape=np.shape(1)):
    return np.random.normal(mean, std, shape)


def add_bg_gaussian_noise(sig, snr):
    sig_noisy = np.array(sig, copy=True)
    mu_sig = np.mean(sig)
    std_sig = np.std(sig)
    mu_noise = mu_sig
    std_noise = std_sig/(10**(snr/20))

    noise = np.random.normal(mu_noise, std_noise, np.shape(sig_noisy))
    sig_noisy = np.add(sig_noisy, noise)

    return sig_noisy


def add_transcient_noise_filtered(sig, snr, noise_len, mu_noise):
    sig_noisy = np.array(sig, copy=True)
    idx_start = int(np.random.rand() * len(sig))
    noise_len = int((np.random.normal() + 1)*noise_len)

    if idx_start + noise_len > len(sig):
        idx_range = range(idx_start, len(sig) - 1)
    else:
        idx_range = range(idx_start, idx_start + noise_len - 1)

    std_sig = np.std(sig)
    std_noise = std_sig / (10 ** (snr / 20))
    noise = np.random.normal(mu_noise, std_noise, np.shape(sig_noisy))
    for idx in range(5, len(sig)-5):
        noise[idx] = np.mean(noise[idx-5:idx+5])

    for idx in idx_range:
        sig_noisy[idx] += noise[idx]

    return sig_noisy

def add_step_noise(sig, mu_sig, std_sig):
    sig_noisy = np.array(sig, copy=True)
    idx_start = int(np.random.rand()*len(sig))
    mu_noise = np.random.normal(mu_sig, std_sig*5)
    noise = mu_noise*np.ones(np.shape(sig_noisy))
    for idx in range(idx_start, len(sig_noisy)):
        sig_noisy[idx] += noise[idx]
    return sig_noisy

def add_transcient_noise(sig, snr, noise_len, mu_noise):
    sig_noisy = np.array(sig, copy=True)
    idx_start = int(np.random.rand()*len(sig))
    noise_len = int(np.random.rand()*noise_len)

    if idx_start + noise_len > len(sig):
        idx_range = range(idx_start, len(sig)-1)
    else:
        idx_range = range(idx_start, idx_start+noise_len-1)

    std_sig = np.std(sig)
    std_noise = std_sig / (10 ** (snr / 20))
    noise = np.random.normal(mu_noise, std_noise, np.shape(sig_noisy))

    for idx in idx_range:
        sig_noisy[idx] += noise[idx]

    return sig_noisy
