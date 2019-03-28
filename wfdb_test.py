from wfdb_tool import prepare_training_set
import matplotlib.pyplot as plt

'''lib loading error prevention'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''scientific packages'''
import numpy as np
import pickle
import datetime

prepare_training_set(5000)


