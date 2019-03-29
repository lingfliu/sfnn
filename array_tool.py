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


'''test code'''
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
