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


fs = 500  # sampling rate of mitdb

'''load tmp data'''
(input, label_raw, label_typ_raw) = pickle.load(open('aha_500_qrs.dat', 'rb'))
(label, x_train, y_train, input_dim, output_dim) = pickle.load(open('aha_500_train_fcn.dat', 'rb'))

'''test plot'''
# plt.figure(1)
# plt.ion()
# for ii in range(np.shape(input)[0]):
#     plt.cla()
#     plt.plot(x_train[ii])
#     plt.plot(y_train[ii])
#     plt.plot(label[ii])
#     plt.pause(0.5)
#     # plt.ioff()
#     plt.show()

# hyper params
batch_size = 40
epochs = 2
filter_size = 64
kernel_size = 4
dropout = 0.2

timestep = len(x_train[0])
x_train = np.array(x_train)
y_train = np.array(y_train)

'''build neural'''
input = Input(shape=(input_dim,1))
classifier = input

'''U-NET-FCN'''
'''==downsampling=='''
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

'''U-bottom layer'''
btm = Conv1D(filter_size*8, kernel_size, strides=1, padding='same', activation='relu')(o3)
btm = Conv1D(filter_size*8, kernel_size, strides=1, padding='same', activation='relu')(o3)
btm = Dropout(0.2)(btm)

'''==upsampling=='''
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

classifier = Conv1D(2, 1, activation = 'sigmoid')(o6)

model = Model(input, classifier)
print(model.summary())

model.compile(optimizer='adam', metrics=['acc', 'mae'], loss='categorical_crossentropy')

'''65% for training, 20% for validation, 15% for testing'''
len_set = np.shape(x_train)[0]
len_train = 4000 # (int)(len_set * 0.65)
len_valid = 1500 # (int)(len_set * 0.2)
len_test = 500
# len_train = (int)(len_set * 0.65)
# len_valid = (int)(len_set * 0.2)
# len_test = len_set - len_train - len_valid

hist = model.fit(x_train[:len_train], y_train[:len_train],
                 validation_data=(x_train[len_train:len_train + len_valid], y_train[len_train:len_train + len_valid]),
                 batch_size=batch_size, epochs=epochs, verbose=1)

tested = x_train[len_train + len_valid:len_train+len_valid+len_test]
expected = y_train[len_train + len_valid:len_train+len_valid+len_test]
predicted = model.predict(tested)


# '''test plot'''
# plt.figure(1)
# plt.ion()
# for ii in range(np.shape(tested)[0]):
#     plt.cla()
#     ex = [np.argmax(p) for p in expected[ii]]
#     pr = [np.argmax(p) for p in predicted[ii]]
#     plt.plot(tested[ii])
#     plt.plot(ex)
#     plt.plot(pr)
#     plt.legend(['sig', 'expect', 'predict'])
#     plt.pause(1)
#     # plt.ioff()
#     plt.show()

# '''save the results'''
# date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
# model.save('aha_500_semantic_seg_fcn' + date_str + '.h5')
# hist_name = 'aha_500_semantic_seg_fcn' + date_str +'.hist'
# pickle.dump(hist, open(hist_name, 'wb'))

