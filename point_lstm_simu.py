from keras.layers import Input, Dropout, Reshape, Dense, Flatten
from keras.layers import Bidirectional, LSTM, ConvLSTM2D, TimeDistributed
from keras.models import Sequential, Model
import keras
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import ecg_simulator
import matplotlib.pyplot as plt


"""create training signal"""
sigs = []
sigs_noisy = []
sample_num = 1000
sample_len = 500

input_dim = 1
batch_size = 20
epochs = 50
timestep = sample_len
filter_size = 50
kernel_size = 2

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
    seq_noisy = np.array([sig[idx:idx+input_dim] for idx in range(len(sig))])
    x_train.append(seq_noisy)
x_train = np.array(x_train)

print(np.shape(x_train))

'''use the middle of the input as the desired output'''
x_train_decoded = []
for sig in sigs:
    seq_decoded = np.array([sig[idx:idx+input_dim] for idx in range(len(sig))])
    x_train_decoded.append(seq_decoded)
x_train_decoded = np.array(x_train_decoded)

"""denoise autoencoder"""
input_sig = Input(shape=(timestep, input_dim))
enc = Bidirectional(LSTM(filter_size, return_sequences=True))(input_sig)
enc = Bidirectional(LSTM(filter_size, return_sequences=True))(enc)
enc = Bidirectional(LSTM(filter_size, return_sequences=True))(enc)
enc = Bidirectional(LSTM(filter_size, return_sequences=True))(enc)

enc = TimeDistributed(Dense(1, activation='linear'))(enc)

dae = Model(input_sig, enc)

dae.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9), loss='mean_squared_error', metrics=['accuracy'])
dae.fit(x_train[:500], x_train_decoded[:500], validation_data=(x_train[500:800], x_train_decoded[500:800]), batch_size=batch_size, epochs=epochs, verbose=1)


predict_idx = 800
x_dec = dae.predict(x_train[predict_idx:])


for m in range(2):
    for n in range(2):
        idx = n*2+m
        if idx >= np.shape(x_train)[0]-predict_idx:
            break
        ax = plt.subplot2grid((2,2),(m,n))
        ax.plot(x_dec[idx])
        ax.plot(sigs[idx+predict_idx][input_dim:])
        ax.plot(sigs_noisy[idx+predict_idx][input_dim:])
plt.show()

print(dae.summary())
