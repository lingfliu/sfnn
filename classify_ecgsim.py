from keras.layers import ConvLSTM2D, Dense, Conv1D, TimeDistributed, BatchNormalization, MaxPooling2D, MaxPooling1D
from keras.layers import Bidirectional, CuDNNLSTM, Dropout, LSTM, Add, Conv2D, Multiply
from keras.layers import Reshape, Input, Flatten, BatchNormalization
from keras.models import Model
from keras.utils import  to_categorical

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
batch_size = 20
epochs = 100
filter_size = 80
kernel_size = 4
dropout = 0.2

# stagging the signal
x_train = []
for sig in sigs_noisy:
    seq_noisy = np.array([sig[i*stride:i*stride+input_dim] for i in range((len(sig)-input_dim)//stride)])
    x_train.append(seq_noisy)

labels = []
for idxx in idx:
    label = np.ones(np.shape(idxx))
    label_start = -1
    for m in range(len(idxx)):
        if idxx[m] > 0:
            if idxx[m] == 4:
                label_start = -1
            else:
                label_start = idxx[m]
        else:
            pass

        label[m] = label_start

    labels.append(label)




y_train = []
for label in labels:
    y = np.array([label[i*stride+input_dim//2-output_dim//2:i*stride+input_dim//2 - output_dim//2 + output_dim] for i in range( (len(label)-input_dim)//stride )])
    y = to_categorical(y, num_classes=6)
    y_train.append(y)

# update the timestep
timestep = len(x_train[0])
x_train = np.array(x_train)
y_train = np.array(y_train)

'''build neural'''

input = Input(shape=(timestep, input_dim))
classifier = input


'''ConvNN before putting into LSTM'''
if input_dim > kernel_size:
    classifier = Reshape(target_shape=(timestep, input_dim, 1))(classifier)
    classifier = TimeDistributed(Conv1D(16, kernel_size=kernel_size, data_format='channels_last', activation='relu'))(classifier)
    classifier = TimeDistributed(Conv1D(32, kernel_size=kernel_size, data_format='channels_last', activation='relu'))(classifier)
    classifier = TimeDistributed(Flatten(data_format='channels_last'))(classifier)

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
o1 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(classifier)
o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
o2 = Add()([o1, o2])
o3 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o2)
o3 = Add()([o1, o2, o3])

'''attention model'''
o4 = TimeDistributed(Dense(filter_size*2, activation='softmax'))(o3)
o5 = Multiply()([o3, o4])

classifier = o5

#
# o3 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(Add()([o1, o2]))
# o4 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o3)


''' fully connected DAE'''
# dae = Add()([o1, o2])
# dae = Add()([o3, o4])



if input_dim > kernel_size:
    classifier = TimeDistributed(Dense(160, activation='linear'))(classifier)

classifier = TimeDistributed(Dense(filter_size*2, activation='relu'))(classifier)
classifier = TimeDistributed(Dense(6, activation='softmax'))(classifier)

model = Model(input, classifier)

print(model.summary())
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), metrics=['accuracy'], loss='categorical_crossentropy')

hist = model.fit(x_train[:300], y_train[:300], validation_data=(x_train[300:400], y_train[300:400]), batch_size=batch_size, epochs=epochs, verbose=1)

predicted = model.predict(x_train[400:])
expected = y_train[400:]

# '''save the result'''
date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model.save('classify_ecgsim' + date_str + '.h5')
hist_name = 'classify_ecgsim_hist_' + date_str +'.dat'
pickle.dump(hist, open(hist_name, 'wb'))


# import matplotlib.pyplot as plt
# plot_idx = 20
# plt.plot(predicted[plot_idx])
# plt.plot(expected[plot_idx])
# plt.plot(x_train[plot_idx])
