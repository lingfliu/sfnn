import wfdb
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import shutil

dbs = wfdb.get_dbs()

for db in dbs:
    print(db)

# Demo 1 - Read a wfdb record using the 'rdrecord' function into a wfdb.Record object.
# Plot the signals, and show the data.
record = wfdb.rdrecord('sample-data/a103l')
wfdb.plot_wfdb(record=record, title='Record a103l from Physionet Challenge 2015')
display(record.__dict__)


# Can also read the same files hosted on Physiobank https://physionet.org/physiobank/database/
# in the challenge/2015/training/ database subdirectory. Full url = https://physionet.org/physiobank/database/challenge/2015/training/
record2 = wfdb.rdrecord('a103l', pb_dir='challenge/2015/training/')