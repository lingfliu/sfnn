import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, LSTM, RepeatVector, Convolution1D, MaxPooling1D, UpSampling1D
from tensorflow.python.keras.models import Model

np.random.seed(1337)

n_batch = 51
n_epoch = 110

timesteps = 100
input_dim = 1
latent_dim = 30


def make_sin(period, phase, amp, length):
    '''values between zero and one and centered at .5'''
    assert 0 < amp <= .5

    xs = np.arange(0, length).astype(np.float)
    xs *= np.pi * 2 / period
    return .5 + (amp * np.sin(xs - phase))


def make_random_batch(size):
    length = timesteps
    result = np.empty((size, length, 1), dtype=np.float)
    for i in range(size):
        period = random.random() * 5
        phase = 0  # random.random() * 2*np.pi
        amp = random.random() * .4 + .1
        result[i, :, 0] = make_sin(period, phase, amp, length)
    return result


def make_lstm():
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    sequence_autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    return sequence_autoencoder, encoder


sequence_autoencoder, encoder = make_lstm()
print(sequence_autoencoder.summary())

for epoch in range(n_epoch):
    x_train = make_random_batch(n_batch)
    sequence_autoencoder.fit(x_train, x_train, epochs=1, steps_per_epoch=10, verbose=2)

predicted = sequence_autoencoder.predict(x_train[0:1])
expected = x_train[0:1]

for x, y in zip(expected[0], predicted[0]):
    print (x, y)