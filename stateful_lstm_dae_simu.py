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

"""create training signal"""
sigs = []
sigs_noisy = []
sample_num = 1
sample_len = 1000000

input_dim = 200
batch_size = 10
timestep = sample_len-input_dim
filter_size = 100
kernel_size = 2

'''generate noisy rythmic signal'''
sig = ecg_simulator.simu_rythm_sig(sample_len, 0, 1)
sig_noisy = ecg_simulator.add_bg_gaussian_noise(sig, 1, -5)
for i in range(10):
    sig_noisy = ecg_simulator.add_transcient_noise(sig_noisy, 1, 0, 10, np.random.random())

x_train = np.array([sig[idx:idx+input_dim] for idx in range(len(sig)-input_dim)])
'''use the middle of the input as the desired output'''
x_train_decoded = np.array([[sig[idx+input_dim//2]] for idx in range(input_dim//2, len(sig)-input_dim//2)])

batch_idx = 0
x_noisy  = []
x_decoded = []
while(batch_idx < len(x_train)):
    x_noisy.append(x_train[batch_idx:batch_idx+batch_size])
    x_decoded.append(x_train_decoded[batch_idx:batch_idx+batch_size])
    batch_idx += batch_size


"""denoise autoencoder"""

input_sig = Input(shape=(timestep, input_dim))
enc = LSTM(units=filter_size, return_sequences=True, stateful=True, batch_input_shape=(batch_size, timestep, input_dim))(input_sig)
enc = LSTM(units=filter_size, return_sequences=True, stateful=True, batch_input_shape=(batch_size, timestep, filter_size))(enc)
enc = LSTM(units=filter_size, return_sequences=True, stateful=True, batch_input_shape=(batch_size, timestep, filter_size))(enc)
enc = LSTM(units=filter_size, return_sequences=True, stateful=True, batch_input_shape=(batch_size, timestep, filter_size))(enc)
enc = TimeDistributed(Dense(1, activation='relu'))(enc)

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
dae.compile(optimizer='adadelta', loss='logcosh', metrics=['accuracy'])
dae.fit(x_train[:200], x_train_decoded[:200], validation_data=(x_train[200:350], x_train_decoded[200:350]), batch_size=batch_size, epochs=2, verbose=2, shuffle=False)

x_dec = dae.predict(x_train[350:])

for m in range(4):
    for n in range(4):
        idx = n*4+m
        if idx >= 50:
            break
        ax = plt.subplot2grid((4,4),(m,n))
        ax.plot(x_dec[idx])
        ax.plot(sigs[idx+350][input_dim:])
        ax.plot(sigs_noisy[idx+350][input_dim:])
plt.show()
