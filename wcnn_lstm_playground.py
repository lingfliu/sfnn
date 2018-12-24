from keras.layers import Conv1D, MaxPool1D, LSTM, Bidirectional, UpSampling1D
from keras.models import Sequential
import keras
from keras.datasets import  mnist

import ecg_simulator

import numpy as np
from matplotlib import pyplot as plt


from keras.layers import Input, Dense
from keras.models import Model

"""create signal"""
sample_len = 10000

y = []
for m in range(100):
    data = ecg_simulator.simu_rythm_sig(sample_len, 0, 1)
    ecg_simulator.add_bg_gaussian_noise(data, 1, -10)
    for i in range(20):
        ecg_simulator.add_transcient_noise(data, 1, -5, 500, np.random.random())

    y.append(data)



"""denoise autoencoder"""
input_dim = 512
encode_dim = 512
input_sig = Input(shape=(input_dim, ))

enc = Conv1D(32, (20,1), padding='same')(input_sig)
enc = MaxPool1D(2, padding='same')(enc)
enc = Conv1D(64, (20,1), padding='same')(enc)
enc = MaxPool1D(2, padding='same')(enc)

enc = Bidirectional(LSTM(256, activation='relu', return_sequences=True))(enc)
enc = Bidirectional(LSTM(256, activation='relu', return_sequences=True))(enc)
enc = Bidirectional(LSTM(256, activation='relu', return_sequences=True))(enc)

enc = Dense(512, activation='relu')(enc)
enc = Dense(512, activation='relu')(enc)
enc = Dense(512, activation='relu')(enc)
'''enc end here'''

dec = Dense(512, activation='relu')(enc)
dec = Dense(512, activation='relu')(dec)
dec = Dense(512, activation='relu')(dec)


dec = Bidirectional(LSTM(256, activation='relu', return_sequences=True), merge_mode='concate')(dec)
dec = Bidirectional(LSTM(256, activation='relu', return_sequences=True), merge_mode='concate')(dec)
dec = Bidirectional(LSTM(256, activation='relu', return_sequences=True), merge_mode='concate')(dec)

dec = Conv1D(32, (20,1), activation='relu', padding='same')(dec)
dec = UpSampling1D(2)(dec)
dec = Conv1D(32, (20,1), activation='relu', padding='same')(dec)
dec = UpSampling1D(2)(dec)

dae = Model(input_sig, dec)
dae.compile(optimizer='adadelta', loss='mean_squared_error')

#todo: sequential the signal per sample
dae.fit()



sig_dim = 32
input_sig = Input(shape=(784, ))
encoded = Dense(sig_dim, activation='relu')(input_sig)
decoded = Dense(784, activation='sigmoid')(encoded)

dae = Model(input_sig, decoded)

encoder = Model(input_sig, encoded)
decoded_input = Input(shape=(sig_dim, ))
decoder_layer = dae.layers[-1]
decoder = Model(decoded_input, decoder_layer(decoded_input))

dae.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

dae.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))


# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()