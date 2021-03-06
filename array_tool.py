import numpy as np
import matplotlib.pyplot as plt


def resample(sig, fs, fs_new, method='linear'):
    re_sig = np.zeros((int)(len(sig)/fs*fs_new))
    if method == 'linear':
        idx0 = 0
        for idx in range(len(re_sig)-1):
            # by default align the first data
            t = idx/fs_new
            if (idx0+1)/fs > t >= idx0/fs:
                dt_prev = t - idx0/fs
                dt_next = (idx0+1)/fs - t
                re_sig[idx] = (sig[idx0]*dt_next+sig[idx0+1]*dt_prev)*fs
            else:
                while t < idx0/fs or t >= (idx0+1)/fs:
                    idx0 += 1
                    if idx0 >= len(sig):
                        break
                if idx0 >= len(sig)-1:
                    break
                else:
                    dt_prev = t - idx0/fs
                    dt_next = (idx0+1)/fs - t
                    re_sig[idx] = (sig[idx0]*dt_next+sig[idx0+1]*dt_prev)*fs
        return re_sig
    elif method == 'label':
        for idx in range(len(sig)-1):
            re_sig[(int)(idx/fs*fs_new)] = sig[idx]
        return re_sig
    elif method == 'label_str':
        re_sig = ['']*((int)(len(sig)/fs*fs_new))
        for idx in range(len(sig)-1):
            re_sig[(int)(idx/fs*fs_new)] = sig[idx]
        return re_sig

def med_filter(sig, med_len):
    sig_pad = []
    for idx in range(med_len):
        sig_pad.append(sig[0])
    for s in sig:
        sig_pad.append(s)
    for idx in range(med_len):
        sig_pad.append(sig[-1])
    med_sig = [sig_pad[idx] - np.median(sig_pad[idx-med_len:idx+med_len]) for idx in range(med_len, len(sig)+med_len, 1)]
    return med_sig

'''test code'''
# sig = [1,1,3,2,1,5,7, 8, 4, 2, 0]
# med_sig = med_filter(sig, med_len=3)
# a = [(1,[0,0,0]), (3,[1,1,1]), (6,[2,2,2]), (0,[3,3,3])]
# a = sorted(a, key=lambda x: x[0])
# print(a)
#
# sig = [p for p in range(10000)]
# fs = 3
# fs_new = 5
# re_sig = resample(sig, fs, fs_new)
# x = np.linspace(0,100,num=len(sig))
# plt.plot(x, sig)
# x = np.linspace(0,100,num=len(sig)/fs*fs_new)
# plt.plot(x, re_sig)
# plt.show()


#
# sig = np.zeros(shape=1000)
# sig[::5] = 1
# fs = 3
# fs_new = 5
# re_sig = resample(sig, fs, fs_new, method='label')
# x = np.linspace(0,1000,num=len(sig))
# plt.plot(x, sig)
# x = np.linspace(0,1000,num=len(sig)/fs*fs_new)
# plt.plot(x, re_sig)
# plt.show()
