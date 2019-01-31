import wfdb
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os


def download_db_list():
    dbs = wfdb.get_dbs()
    return dbs


def download_db(target_dir=None, db='ahadb'):
    cwd = os.getcwd()
    if not target_dir:
        target_dir = os.path.join(cwd, db)
    wfdb.dl_database(db, dl_dir=target_dir)


if __name__ == '__main__':
    download_db(db='ltdb')
