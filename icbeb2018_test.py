from icbeb_tool import load_icbeb2018
import matplotlib.pyplot as plt

'''lib loading error prevention'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''scientific packages'''
import numpy as np
import pickle
import datetime

load_icbeb2018([1])


