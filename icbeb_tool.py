import scipy.io as sio
import numpy as np
import os


def load_icbeb2019(db_dir='icbeb2019'):
    dat_dir = os.path.join(os.getcwd(), db_dir, 'data')
    ref_dir = os.path.join(os.getcwd(), db_dir, 'ref')
    dat_file_list = []
    for root, dirs, files in os.walk(dat_dir):
        [dat_file_list.append(f) for f in files]

    dat_name = [dat_file.split('.')[0].split('_')[1] for dat_file in dat_file_list]

    ref_file_list = []
    for root, dirs, files in os.walk(ref_dir):
        [ref_file_list.append(f) for f in files]

    ecg_list = []
    label_list = []
    for name in dat_name:
        print('loading data ', name)
        dat_file = 'data_' + name

        mat_data = sio.matlab.loadmat(os.path.join(dat_dir, dat_file))
        ecg = mat_data['ecg']

        ref_file = 'R_' + name
        mat_label = sio.matlab.loadmat(os.path.join(ref_dir, ref_file))
        r_list = mat_label['R_peak']
        label = np.zeros(np.shape(ecg))
        for r in r_list:
            label[r-1] = 1

        ecg_list.append(ecg[:,0])
        label_list.append(label[:,0])

    return (ecg_list, label_list)

