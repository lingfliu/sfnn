'''A neural network checker'''

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv1D, LSTM, TimeDistributed, Reshape
from keras.layers import Conv1D, MaxPooling1D, Input
from keras.models import Model
from keras.datasets import imdb

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

input = Input(shape=(500,200))
nn = LSTM(64, return_sequences=True)(input)
nn = Reshape(target_shape=(500,64,1))(nn)
nn = TimeDistributed(Conv1D(64, kernel_size=20, strides=1, activation='relu'))(nn)
nn = TimeDistributed(MaxPooling1D(pool_size=2))(nn)
nn = TimeDistributed(Flatten(data_format='channels_first'))(nn)
nn = TimeDistributed(Dense(1))(nn)
model = Model(input, nn)

print(model.summary())
