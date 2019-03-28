from keras.utils import to_categorical
from keras.models import load_model

import matplotlib.pyplot as plt

'''lib loading error prevention'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''========================'''
'''tensorflow configuration'''
'''========================'''
import tensorflow as tf
from keras import backend as K
num_cores = 48

num_CPU = 1
num_GPU = 1

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


'''scientific packages'''
import numpy as np
import pickle
import datetime


fs = 500 # sampling rate of mitdb

model_h5 = 'icbeb2019_semantic_seg_2019_03_24_10_28_11.h5'
hist_rec = 'icbeb2019_semantic_seg_2019_03_24_10_28_11.hist'


'''load tmp data'''
(data, label) = pickle.load(open('icbeb_test.tmp', 'rb'))

'''global parameters'''
input_dim = 1
output_dim = 1

if input_dim < output_dim:
    print('input_dim smaller than output_dim, quit task')

stride = output_dim
timestep = 0

# hyper params
batch_size = 40
epochs = 400
filter_size = 80
kernel_size = 4
dropout = 0.2


# stagging the signal
x_train = []
for dat in data:
    seq = np.array([dat[i*stride:i*stride+input_dim] for i in range((len(dat)-input_dim)//stride)])
    x_train.append(seq)

y_train = []
for lb in label:
    y = np.array([lb[i*stride+input_dim//2-output_dim//2:i*stride+input_dim//2 - output_dim//2 + output_dim] for i in range( (len(lb)-input_dim)//stride )])
    y = to_categorical(y, num_classes=2)
    y_train.append(y)

'''load model'''
model = load_model(model_h5)

timestep = len(x_train[0])
x_train = np.array(x_train)
y_train = np.array(y_train)

tested = x_train[1800:]
expected = y_train[1800:]
predicted = model.predict(tested)

ex = [np.argmax(p) for p in expected[2]]
pr = [np.argmax(p) for p in predicted[2]]

plt.plot(tested[2])
plt.plot(ex)
plt.plot(pr)
plt.legend(['signal', 'expected', 'predicted']),
plt.show()
