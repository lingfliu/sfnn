from keras.layers import ConvLSTM2D, Dense, LSTM, Conv1D, BatchNormalization, Reshape, Input, TimeDistributed
from keras.layers import TimeDistributed, MaxPooling1D
from keras.models import Model
import keras

'''lib loading error prevention'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


'''config cpu & gpu'''
import tensorflow as tf
from keras import backend as K
num_cores = 48

num_CPU = 24
num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

import numpy as np
import matplotlib.pyplot as plt
from ecgsyn import ecgsyn
import ecg_simulator

'''global params'''
sample_len = 10
batch_size = 2
epochs = 10
input_dim = 40
output_dim = 3
if input_dim < output_dim:
    print('input_dim smaller than output_dim, quit task')

stride = output_dim
timestep = 400

filter_size = 20
kernel_size = 6

sigs = []
idx = []
sigs_noisy = []


for i in range(sample_len):
    print('generating ecg: ', i)
    (ecg, pqrst_idx) = ecgsyn(sfecg=64, N=10)
    ecg_noisy = ecg_simulator.add_bg_gaussian_noise(ecg, 0, -5)

    for i in range(10):
        ecg_noisy = ecg_simulator.add_transcient_noise(ecg_noisy, 1, 0, 100, np.random.random())
    sigs.append(ecg)
    sigs_noisy.append(ecg_noisy)

    idx.append(pqrst_idx)

x_train = []
for sig in sigs_noisy:

    seq_noisy = np.array([sig[i*stride:i*stride+input_dim] for i in range( (len(sig)-input_dim)//stride )])
    x_train.append(seq_noisy)

y_train = []
for sig in sigs:
    y = np.array([sig[i*stride+input_dim//2-output_dim//2:i*stride+input_dim//2 - output_dim//2 + output_dim] for i in range( (len(sig)-input_dim)//stride )])
    y_train.append(y)

timestep = len(x_train[0])

x_train = np.array(x_train)
y_train = np.array(y_train)

'''build the network'''
input_sig = Input(shape=(timestep, input_dim))
enc = Reshape(target_shape=(timestep, input_dim, 1, 1))(input_sig)
# for _ in range(20):
#     enc = ConvLSTM2D(filters=filter_size, kernel_size=(kernel_size,1), strides=1, return_sequences=True)(enc)
#     enc = BatchNormalization()(enc)

# enc = TimeDistributed(Dense(1, activation='linear'))(enc)

model = Model(input_sig, enc)

print(model.summary())

model.compile(optimizer='adagrad', metrics=['cosine'], loss='mean_squared_error')
model.fit(x_train[:10], y_train[:10], validation_data=(x_train[:10], y_train[:10]), batch_size=batch_size, epochs=epochs, verbose=1)
