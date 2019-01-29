"""this script generate simulated data"""

from multiprocessing import Process, Pool, Manager
import numpy as np
import matplotlib.pyplot as plt
import time
from ecgsyn import ecgsyn
from sig_simulator import add_bg_gaussian_noise, add_transcient_noise, add_transcient_noise_filtered, add_step_noise
import pickle
'''global parameters'''
sample_len = 500
ecg_N = 10
sps = 250
base_snr = 8
artifact_snr = 0
sigs = []
idx = []
sigs_noisy = []


def task_noisy_ecg_generate(i, queue):
    # print('generating ecg of : ', i)
    (ecg, pqrst_idx) = ecgsyn(sfecg=sps, anoise=0, N=ecg_N)  # sampling freq 250, num of heart beats 10
    ecg_noisy = add_bg_gaussian_noise(ecg, base_snr)
    mu_ecg = np.mean(ecg)
    std_ecg = np.std(ecg)
    mu_noise = mu_ecg + np.random.normal(mu_ecg, std_ecg) + mu_ecg
    for _ in range(10):
        ecg_noisy = add_transcient_noise(ecg_noisy, artifact_snr, len(ecg) / 10, mu_noise)

    for _ in range(5):
        ecg_noisy = add_step_noise(ecg_noisy, mu_ecg, std_ecg)

    queue.put((ecg, ecg_noisy, pqrst_idx))


if __name__ == '__main__':

    '''generate synthetic data'''

    '''using multiprocessing to accelerate'''
    queue = Manager().Queue(maxsize=sample_len)
    pool = Pool()
    for i in range(sample_len):
        pool.apply_async(task_noisy_ecg_generate, args=(i, queue))

    pool.close()
    while True:
        (ecg, ecg_noisy, pqrst_idx) = queue.get()
        sigs.append(ecg)
        sigs_noisy.append(ecg_noisy)
        idx.append(pqrst_idx)
        print('sig len=', len(sigs))
        if len(sigs) >= sample_len:
            pickle.dump((sigs, sigs_noisy, idx, sps, ecg_N, base_snr, artifact_snr), open('dae_ecgsim_stepnoise_500.dat', 'wb'))
            break

    pool.join()

    plt.plot(sigs_noisy[0])
    plt.plot(sigs[0])
    plt.show()

