from keras.layers import Conv1D, LSTM, Bidirectional, UpSampling1D, Flatten, ConvLSTM2D, Reshape, MaxPooling2D, MaxPooling1D, MaxPooling3D, UpSampling2D
from keras.layers import TimeDistributed
from keras.models import Sequential
import keras
from keras.datasets import mnist

import ecg_simulator

import numpy as np
from matplotlib import pyplot as plt

import math

from keras.layers import Input, Dense
from keras.models import Model


import tensorflow as tf
from keras import backend as K

num_cores = 96

num_CPU = 2
num_GPU = 1

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

"""create training signal"""
sigs = []
sigs_noisy = []
sample_num = 1000
sample_len = 500

input_dim = 50
batch_size = 10
timestep = sample_len-input_dim
epochs = 50
filter_size = 100
kernel_size = 3

'''generate noisy rythmic signal'''
for m in range(sample_num):
    sig = ecg_simulator.simu_rythm_sig(sample_len, 0, 1)
    sig_noisy = ecg_simulator.add_bg_gaussian_noise(sig, 1, -5)
    for i in range(10):
        sig_noisy = ecg_simulator.add_transcient_noise(sig_noisy, 1, 0, 10, np.random.random())

    sigs.append(sig)
    sigs_noisy.append(sig_noisy)

x_train = []
for sig in sigs_noisy:
    seq_noisy = np.array([sig[idx:idx+input_dim] for idx in range(len(sig)-input_dim)])
    x_train.append(seq_noisy)
x_train = np.array(x_train)

'''use the middle of the input as the desired output'''
x_train_decoded = []
for sig in sigs:
    seq_decoded = np.array([[sig[idx+input_dim//2]] for idx in range(input_dim//2, len(sig)-input_dim//2)])
    x_train_decoded.append(seq_decoded)
x_train_decoded = np.array(x_train_decoded)
# x_train_decoded.reshape(200,68,1)

"""denoise autoencoder"""


input_sig = Input(shape=(timestep, input_dim))
enc = Reshape(target_shape=(timestep, input_dim,1,1))(input_sig)
# enc = ConvLSTM2D(filter_size, kernel_size=(kernel_size,1), strides=1, return_sequences=True)(enc)
# enc = ConvLSTM2D(filter_size, kernel_size=(kernel_size,1), strides=1, return_sequences=True)(enc)

enc = LSTM(units=filter_size, return_sequences=True)(input_sig)
enc = LSTM(units=filter_size, return_sequences=True)(enc)
enc = Reshape(target_shape=(timestep,filter_size, 1))(enc)
enc = TimeDistributed(Conv1D(64,kernel_size=kernel_size, strides=1, activation='relu'))(enc)
enc = TimeDistributed(MaxPooling1D(pool_size=2))(enc)
enc = TimeDistributed(Conv1D(64,kernel_size=kernel_size, strides=1, activation='relu'))(enc)
enc = TimeDistributed(MaxPooling1D(pool_size=2))(enc)
enc = TimeDistributed(Flatten(data_format='channels_first'))(enc)
enc = TimeDistributed(Dense(1, activation='linear'))(enc)

'''pooling over each input'''
# enc = Reshape(target_shape=(timestep, (input_dim-kernel_size*2)*filter_size*2))(enc)
# enc = Reshape(target_shape=(input_dim-2, 64, timestep))(enc)
# enc = MaxPooling2D(pool_size=(2,1))(enc)
# enc = Reshape(target_shape=(timestep, (input_dim-2)//2*64))(enc)

'''calculate the output for each timestep'''
# enc = TimeDistributed(Dense(240, activation='relu'))(enc)
# enc = TimeDistributed(Dense(120, activation='relu'))(enc)
# enc = TimeDistributed(Dense(1, activation='relu'))(enc)

'''end of enc'''

dae = Model(input_sig, enc)

print(dae.summary())


# #todo: sequential the signal per sample
# # y is a array of signal sequences
dae.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9), loss='mean_squared_error', metrics=['cosine'])
dae.fit(x_train[:500], x_train_decoded[:500], validation_data=(x_train[500:800], x_train_decoded[500:800]), batch_size=batch_size, epochs=epochs, verbose=1)

predict_idx = 800
x_dec = dae.predict(x_train[predict_idx:])

for m in range(4):
    for n in range(4):
        idx = n*4+m
        if idx >= 50:
            break
        ax = plt.subplot2grid((4,4),(m,n))
        ax.plot(x_dec[idx])
        ax.plot(sigs[idx+predict_idx][input_dim:])
        ax.plot(sigs_noisy[idx+predict_idx][input_dim:])
plt.show()
