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

"""
# 将wfdb数据格式化
"""
def format_data(wfdb_data):
    data = [Ecg]
    multi_ecg = MultiEcg(data)

    return multi_ecg

"""
# 将wfdb标签格式化
"""
def format_label(wfdb_data, wfdb_label):
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

if __name__ == '__main__':
    dl_all_db()
