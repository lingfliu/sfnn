from keras.layers import ConvLSTM2D, Dense, Conv1D, TimeDistributed, BatchNormalization, MaxPooling2D, MaxPooling1D
from keras.layers import Bidirectional, CuDNNLSTM, Dropout, LSTM, Add, Conv2D, Multiply
from keras.layers import Reshape, Input, Flatten, BatchNormalization
from keras.models import Model
import keras

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

'''load data'''
(sigs, sigs_noisy, idx, sps, ecg_N, base_snr, artifact_snr) = pickle.load(open('dae_ecgsim_stepnoise_500.dat', 'rb'))

'''global parameters'''
sample_len = len(sigs)
input_dim = 1
output_dim = 1

if input_dim < output_dim:
    print('input_dim smaller than output_dim, quit task')

stride = output_dim
timestep = 0

# neural params
batch_size = 40
epochs = 240
filter_size = 80
kernel_size = 4
dropout = 0.2

# stagging the signal
x_train = []
for sig in sigs_noisy:
    seq_noisy = np.array([sig[i*stride:i*stride+input_dim] for i in range((len(sig)-input_dim)//stride)])
    x_train.append(seq_noisy)

y_train = []
for sig in sigs:
    y = np.array([sig[i*stride+input_dim//2-output_dim//2:i*stride+input_dim//2 - output_dim//2 + output_dim] for i in range( (len(sig)-input_dim)//stride )])
    y_train.append(y)

# update the timestep
timestep = len(x_train[0])
x_train = np.array(x_train)
y_train = np.array(y_train)

'''build neural'''

input = Input(shape=(timestep, input_dim))
dae = input


'''ConvNN before putting into LSTM'''
if input_dim > kernel_size:
    dae = Reshape(target_shape=(timestep, input_dim, 1))(dae)
    dae = TimeDistributed(Conv1D(16, kernel_size=kernel_size, data_format='channels_last', activation='relu'))(dae)
    dae = TimeDistributed(Conv1D(32, kernel_size=kernel_size, data_format='channels_last', activation='relu'))(dae)
    dae = TimeDistributed(Flatten(data_format='channels_last'))(dae)

'''residual LSTM'''
# layer_input = []
# layer_output = []
# for i in range(3):
#     if len(layer_output) <= 0:
#         ii = dae
#     elif len(layer_output) == 1:
#         ii = layer_output[0]
#     else:
#         ii = Add()(layer_output[:i])
#
#     layer_input.append(ii)
#     oo = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(ii)
#     layer_output.append(oo)
#
# dae = layer_output[-1]

'''manually partially connected residual LSTM'''
# i1 = dae
# o1 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(i1)
# i2 = o1
# o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(i2)
# i3 = Add()([o1, o2])
# o3 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(i3)
# i4 = o3
# o4 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(i4)
#
# dae = Add()([o3, o4])


'''LSTM'''
# o1 = (CuDNNLSTM(filter_size, return_sequences=True))(dae)
# o2 = (CuDNNLSTM(filter_size, return_sequences=True))(o1)


'''bidirectional LSTM'''
o1 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(dae)
o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
o2 = Add()([o1, o2])
o2 = Dropout(0.2)(o2)
o2 = BatchNormalization()(o2)
o3 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o2)
o3 = Add()([o1, o2, o3])
o3 = Dropout(0.2)(o3)
o3 = BatchNormalization()(o3)

# '''attention model'''
# o3a = TimeDistributed(Dense(filter_size*2, activation='softmax'))(o3)
# o3v = Multiply()([o3a, o1])

o4 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o3)
o4 = Add()([o3, o4])
o4 = Dropout(0.2)(o4)
o4 = BatchNormalization()(o4)

o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o4)
o5 = Add()([o3, o4, o5])
o5 = Dropout(0.2)(o5)
o5 = BatchNormalization()(o5)

o6 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)

'''attention model'''
o6 = TimeDistributed(Dense(filter_size*2, activation='relu'))(o6)
o6 = TimeDistributed(Dense(filter_size*2, activation='softmax'))(o6)

o6v = Multiply()([o6, o5])

dae = o6v

#
# o3 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(Add()([o1, o2]))
# o4 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o3)


''' fully connected DAE'''
# dae = Add()([o1, o2])
# dae = Add()([o3, o4])



if input_dim > kernel_size:
    dae = TimeDistributed(Dense(160, activation='linear'))(dae)

dae = TimeDistributed(Dense(filter_size*2, activation='relu'))(dae)
dae = TimeDistributed(Dense(1, activation='linear'))(dae)

model = Model(input, dae)

print(model.summary())
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), metrics=['mae'], loss='logcosh')

hist = model.fit(x_train[:300], y_train[:300], validation_data=(x_train[300:400], y_train[300:400]), batch_size=batch_size, epochs=epochs, verbose=1)

predicted = model.predict(x_train[400:])
expected = y_train[400:]

# '''save the result'''
date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model.save('dae_ecgsim' + date_str + '.h5')
hist_name = 'dae_ecgsim_hist_' + date_str +'.dat'
pickle.dump(hist, open(hist_name, 'wb'))


import matplotlib.pyplot as plt
plot_idx = 30
plt.plot(predicted[plot_idx])
plt.plot(expected[plot_idx])
plt.plot(x_train[plot_idx])
