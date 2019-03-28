from wfdb_tool import prepare_training_set

import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Add, Multiply, Dense, Conv1D, Concatenate, MaxPooling1D, Reshape, UpSampling1D
from keras.layers import Input, BatchNormalization
from keras.layers import Dropout

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

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                        inter_op_parallelism_threads=num_cores, \
                        allow_soft_placement=True, \
                        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True

session = tf.Session(config=config)
K.set_session(session)


'''scientific packages'''
import numpy as np
import pickle
import datetime


fs = 360  # sampling rate of mitdb

'''load tmp data'''
(input, label_raw, label) = pickle.load(open('wfdb_qrs.tmp', 'rb'))

'''global parameters'''
input_dim = 5000
output_dim = 5000

if input_dim < output_dim:
    print('input_dim smaller than output_dim, quit task')

stride = output_dim
timestep = 0

# hyper params
batch_size = 40
epochs = 100
filter_size = 64
kernel_size = 4
dropout = 0.2

# stagging the signal
x_train = []
for dat in input:
    dat = np.reshape(dat, [len(dat), 1])
    x_train.append(dat)

y_train = []
for lb in label:
    lb_1 = np.reshape(lb, [len(lb), 1])
    lb_1 = to_categorical(lb_1, num_classes=2)
    y_train.append(lb_1)

timestep = len(x_train[0])
x_train = np.array(x_train)
y_train = np.array(y_train)

'''build neural'''
input = Input(shape=(input_dim,1))
classifier = input

'''U-NET-FCN'''
'''==encoder=='''
o1 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(classifier)
o1 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(classifier)
o1 = Dropout(0.2)(o1)
o1 = BatchNormalization()(o1)
o1_norm = o1
o1 = MaxPooling1D(pool_size=2)(o1)

o2 = Conv1D(filter_size*2, kernel_size, strides=1, padding='same', activation='relu')(o1)
o2 = Conv1D(filter_size*2, kernel_size, strides=1, padding='same', activation='relu')(o1)
o2 = Dropout(0.2)(o2)
o2 = BatchNormalization()(o2)
o2_norm = o2
o2 = MaxPooling1D(pool_size=2)(o2)

o3 = Conv1D(filter_size*4, kernel_size, strides=1, padding='same', activation='relu')(o2)
o3 = Conv1D(filter_size*4, kernel_size, strides=1, padding='same', activation='relu')(o2)
o3 = Dropout(0.2)(o3)
o3 = BatchNormalization()(o3)
o3_norm = o3
o3 = MaxPooling1D(pool_size=2)(o3)

'''U bottom layer'''
btm = Conv1D(filter_size*8, kernel_size, strides=1, padding='same', activation='relu')(o3)
btm = Conv1D(filter_size*8, kernel_size, strides=1, padding='same', activation='relu')(o3)
btm = Dropout(0.2)(btm)

'''==decoder=='''
o4 = UpSampling1D(size=2)(btm)
o4 = Conv1D(filter_size*4, kernel_size, strides=1, padding='same', activation='relu')(o4)
o4 = Conv1D(filter_size*4, kernel_size, strides=1, padding='same', activation='relu')(o4)
o4 = Concatenate()([o3_norm, o4])
o4 = Dropout(0.2)(o4)
o4 = BatchNormalization()(o4)

o5 = UpSampling1D(size=2)(o4)
o5 = Conv1D(filter_size*2, kernel_size, strides=1, padding='same', activation='relu')(o5)
o5 = Conv1D(filter_size*2, kernel_size, strides=1, padding='same', activation='relu')(o5)
o5 = Concatenate()([o2_norm, o5])
o5 = Dropout(0.2)(o5)
o5 = BatchNormalization()(o5)

o6 = UpSampling1D(size=2)(o5)
o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same', activation='relu')(o6)
o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same', activation='relu')(o6)
o6 = Concatenate()([o1_norm,o6])
o6 = Dropout(0.2)(o6)
o6 = BatchNormalization()(o6)

classifier = Conv1D(2, 1, activation = 'softmax')(o6)

model = Model(input, classifier)
print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), metrics=['acc', 'mae'], loss='categorical_crossentropy')

hist = model.fit(x_train[:4500], y_train[:4500], validation_data=(x_train[4500:6000], y_train[4500:6000]), batch_size=batch_size, epochs=epochs, verbose=1)


tested = x_train[6000:]
expected = y_train[6000:]
predicted = model.predict(tested)

# '''save the results'''
date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model.save('wfdb_semantic_seg_fcn' + date_str + '.h5')
hist_name = 'wfdb_semantic_seg_fcn' + date_str +'.hist'
pickle.dump(hist, open(hist_name, 'wb'))

