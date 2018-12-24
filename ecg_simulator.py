
import math
import numpy as np
from matplotlib import pyplot as plt


def generate_sig(inphase_idx, W, T, amp, baseline, y, idx, x_len):
    for x in range(inphase_idx, T):
        if x < T/2-W/2 or x > T/2+W/2:
            y[idx] = baseline
        else:
            d = W/2 - math.fabs(T/2-x)
            y[idx] = d/(W/2)*(amp+baseline)

        idx += 1

        if idx >= x_len:
            break

    return idx

# create triangular rythmic signals given random width and duty ratios
def simu_rythm_sig(x_len, baseline=0, amp=1):

    W = 20 #width of the peak
    DR = 4 #duty ratio of the period

    init_phase = np.random.rand()

    y = np.zeros(x_len)

    idx = 0


    W_rand = math.floor(math.fabs(np.random.rand()*10)+W)
    D_rand = (np.random.rand()*2+1)*DR
    T_rand = math.floor(D_rand*W_rand)
    amp_rand = ((np.random.rand()+0.5)*amp)



    # padding the beginning by phase
    if init_phase > 0:
        inphase_idx = math.floor(T_rand*init_phase)

        idx = generate_sig(inphase_idx, W_rand, T_rand, amp_rand, baseline, y, idx, x_len)

    while idx < x_len:

        W_rand = math.floor(math.fabs(np.random.rand()*10)+W)
        D_rand = (np.random.rand()*2+1)*DR
        T_rand = math.floor(D_rand*W_rand)
        amp_rand = ((np.random.rand()+0.5)*amp)

        inphase_idx = 0

        print (W_rand, D_rand, T_rand, amp_rand)

        idx = generate_sig(inphase_idx, W_rand, T_rand, amp_rand, baseline, y, idx, x_len)

    return y

def add_bg_gaussian_noise(y_array, amp, snr):
    for idx in range(len(y_array)):
        y_array[idx] += np.random.random()*(10**(snr/20*amp))

    return y_array


def add_transcient_noise(y_array, amp, snr, noise_len, noise_baseline):
    idx_start = math.floor(np.random.rand()*len(y_array))
    noise_len = math.floor(np.random.rand()*noise_len)

    if idx_start + noise_len > len(y_array):
        idx_range = range(idx_start, len(y_array)-1)
    else:
        idx_range = range(idx_start, idx_start+noise_len-1)

    for idx in idx_range:
        y_array[idx] += np.random.random()*(10**(snr/20*amp))*noise_baseline

    return y_array



#
# y = simu_rythm_sig(3000, 0, 1)
#
# add_bg_gaussian_noise(y, 1, -10)
#
# for i in range(10):
#     add_transcient_noise(y, 1, -5, 500, np.random.random())
#
# plt.plot(y)
#
# plt.show()


