import wfdb
from wfdb import processing
from ecg_model import MultiEcg, Ecg
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from array_tool import resample
import shutil
import os

def detect_qrs(sig, fs):
    # qrs_idx = processing.xqrs_detect(sig, fs) # using xqrs algorithm
    qrs_idx = processing.qrs.gqrs_detect(sig, fs) # using gqrs algorithm
    return qrs_idx

def format_data(wfdb_data):
    data = [Ecg]
    multi_ecg = MultiEcg(data)
    return multi_ecg

"""
format annotations
"""
def format_anno(wfdb_data, wfdb_label):
    return None

def _gen_default_dbdir():
    return os.path.join(os.getcwd(), 'wfdb')

def dl_all_db(db_dir=None):
    if not db_dir:
        db_dir = _gen_default_dbdir()

    dbs = wfdb.get_dbs()
    for db in dbs:
        db_name = db[0]
        print('downloading db: ', db[0], ' ', db[1])
        wfdb.dl_database(db_name+'/', dl_dir=os.path.join(db_dir, db_name))

def dl_dbs(dbs, db_dir=None):
    if not db_dir:
        db_dir = _gen_default_dbdir()

    for db in dbs:
        wfdb.dl_database(db+'/', dl_dir=os.path.join(db_dir, db))


def load_database(database, db_dir):

    dir = os.path.join(os.getcwd(), db_dir, database)
    file_list = []
    for root, dirs, files in os.walk(dir):
        [file_list.append(f) for f in files]

    dat_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1]=='dat', file_list)]
    hea_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1]=='hea', file_list)]
    atr_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1]=='atr', file_list)]

    record_list = [dat for dat in filter(lambda x:x in hea_list and x in atr_list, dat_list)]

    data_list = []
    anno_list = []
    anno_typ_list = []
    for rec in record_list:
        print('loading data', rec)
        data = wfdb.rdrecord(os.path.join(dir, rec))
        anno = wfdb.rdann(os.path.join(dir, rec), 'atr')

        sig = data.p_signal[:,0]
        anno_idx = anno.sample
        anno_typ = anno.subtype

        data_list.append(sig)
        anno_list.append(anno_idx)
        anno_typ_list.append(anno_typ)

        # '''test code'''
        # anno_typ = np.zeros(np.shape(data.p_signal[:,0]))
        # for idx in range(len(anno.sample)):
        #     anno_typ[anno.sample[idx]] = anno.subtype[idx]
        # plt.plot(data.p_signal[:,0])
        # plt.plot(anno_idx)
        # plt.plot(anno_typ)
        # plt.show()
        # plt.legend(['signal', 'anno_idx', 'anno_typ'])

    return (data_list, anno_list, anno_typ_list)

def prepare_training_set(set_len=5000):
    (data_list, anno_list, anno_typ_list) = load_database('mitdb', db_dir='wfdb')

    input = []
    label = []
    typ = []
    zipped = zip(data_list, anno_list, anno_typ_list)

    cnt = 0
    for (data, anno, anno_typ) in zipped:
        cnt += 1
        anno_idx = np.zeros(np.shape(data))
        anno_typ_idx = ['']*len(data)
        for idx in anno:
            anno_idx[idx] = 1
        for idx in range(len(anno)):
            anno_typ_idx[anno[idx]] = anno_typ[idx]

        ''' resample before splitting'''
        print('resampling ', cnt)
        data = resample(data, 360, 500, method='linear')
        anno_idx = resample(anno_idx, 360, 500, method='label')
        anno_typ_idx = resample(anno_typ_idx, 360, 500, method='label_str')

        idx = 0
        while idx+set_len < len(data):
            d = data[idx:idx+set_len]
            input.append(d)

            lb = anno_idx[idx:idx+set_len]
            label.append(lb)

            at = anno_typ_idx[idx:idx+set_len]
            typ.append(at)

            idx += set_len

    return (input, label, typ)

def prepare_training_set_aha(set_len=5000, db_dir='wfdb/aha'):
    dir = os.path.join(os.getcwd(), db_dir)
    file_list = []
    for root, dirs, files in os.walk(dir):
        [file_list.append(f) for f in files]

    dat_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1]=='dat', file_list)]
    hea_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1]=='hea', file_list)]
    atr_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1]=='atr', file_list)]

    record_list = [dat for dat in filter(lambda x:x in hea_list and x in atr_list, dat_list)]
    data_list = []
    anno_list = []
    anno_typ_list = []
    for rec in record_list:
        print('loading data', rec)
        data = wfdb.rdrecord(os.path.join(dir, rec))
        anno = wfdb.rdann(os.path.join(dir, rec), 'atr')

        sig = data.p_signal
        # use local peaks to trace the R peak
        peaks = processing.find_local_peaks(sig[:,0], 10)

        idx2 = 0
        anno_r = []
        for idx in anno.sample:
            idx_start = idx2
            for idx_peak in peaks[idx_start:]:
                idx2 += 1
                # find the nearest R peak after the QRS onset
                if idx_peak > idx:
                    anno_r.append(idx_peak)
                    break

        anno_r_idx = np.zeros(len(sig[:,0]))
        for idx in anno_r:
            anno_r_idx[idx] = 1
        anno_typ = anno.subtype
        anno_typ_idx = [''] * len(sig)
        for idx in range(len(anno_r)):
            anno_typ_idx[anno_r[idx]] = anno_typ[idx]

        # trunc the data
        sig = sig[anno_r[0]-20:]
        anno_r_idx = anno_r_idx[anno_r[0]-20:]
        anno_r = anno_r - (anno_r[0] + 20)
        anno_typ_idx = anno_typ_idx[anno_r[0]-20:]

        # resample to 500Hz
        sig = resample(sig[:,0], 250, 500, method='linear')
        anno_r_idx = resample(anno_r_idx, 250, 500, method='label')
        anno_typ_idx = resample(anno_typ_idx, 250, 500, method='label_str')

        '''test code'''
        # plt.plot(sig)
        # plt.plot(anno_r_idx)
        # plt.show()

        data_list.append(sig)
        anno_list.append(anno_r_idx)
        anno_typ_list.append(anno_typ_idx)

    input = []
    label = []
    typ = []
    zipped = zip(data_list, anno_list, anno_typ_list)
    cnt = 0
    for (data, anno_r_idx, anno_typ_idx) in zipped:
        cnt += 1
        idx = 0
        while idx+set_len < len(data):
            d = data[idx:idx + set_len]
            input.append(d)

            lb = anno_r_idx[idx:idx + set_len]
            label.append(lb)

            at = anno_typ_idx[idx:idx + set_len]
            typ.append(at)

            idx += set_len

    return (input, label, typ)

