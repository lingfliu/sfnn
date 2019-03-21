import wfdb
from wfdb import  processing
from ecg_model import MultiEcg, Ecg
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
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
    for rec in record_list:
        print('loading data', rec)
        data = wfdb.rdrecord(os.path.join(dir, rec))
        anno = wfdb.rdann(os.path.join(dir, rec), 'atr')

        data_list.append(data.p_signal[:,0])
        anno_list.append(anno.sample)

    return (data_list, anno_list)

def prepare_training_set(set_len=5000):
    (data_list, anno_list) = load_database('mitdb', db_dir='wfdb')

    input = []
    label = []
    zipped = zip(data_list, anno_list)
    for (data, anno) in zipped:
        idx = 0
        while idx < len(data):
            d = data[idx:idx+set_len]
            input.append(d)

            lb = []
            for idx2 in range(len(d)):
                lb.append(0)

            anno_idx = [ai for ai in filter(lambda x:x-idx<set_len and x-idx>=0, anno)]

            for idx2 in anno_idx:
                lb[idx2-idx] = 1

            label.append(lb)

            idx += set_len

    return (input, label)

#
# def main():
#     (input, label) = prepare_training_set()
#
#
# if __name__ == '__main__':
#
#     # dl_all_db()
#     # dl_dbs(['ltdb'])
#     # data = '210'
#     # record = wfdb.rdrecord('wfdb/mitdb/'+data)
#     # anno = wfdb.rdann('wfdb/mitdb/'+data, 'atr')
#     # wfdb.plot_wfdb(record=record, annotation=anno, title='mitdb-100')
#     main()

