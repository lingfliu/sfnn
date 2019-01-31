from keras.layers import ConvLSTM2D, Dense, Conv1D, TimeDistributed, BatchNormalization, MaxPooling2D, MaxPooling1D
from keras.layers import Bidirectional, CuDNNLSTM, Dropout, LSTM, Add, Conv2D, Multiply
from keras.layers import Reshape, Input, Flatten, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical

import keras
import matplotlib.pyplot as plt

'''lib loading error prevention'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
epochs = 200
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
        if idxx[m] < 0:
            # search the first non-negative idxx
            continue
        else:
            # label the precedent points
            label_start = idxx[m] - 1
            break

    for m in range(len(idxx)):
        if idxx[m] >= 0:
            if idxx[m] == 4:
                label_start = -1
            else:
                label_start = idxx[m]
        else:
            pass

        if idxx[m] == 4:
            label[m] = idxx[m]
        else:
            label[m] = label_start

    labels.append(label)

truth = []
for sig in sigs:
    tr = np.array([sig[i*stride+input_dim//2-output_dim//2:i*stride+input_dim//2 - output_dim//2 + output_dim] for i in range( (len(sig)-input_dim)//stride )])
    truth.append(tr)

label_train = []
labels = idx
for label in labels:
    y = np.array([label[i*stride+input_dim//2-output_dim//2:i*stride+input_dim//2 - output_dim//2 + output_dim] for i in range( (len(label)-input_dim)//stride )])
    y = to_categorical(y, num_classes=6)
    label_train.append(y)

'''data test code'''
# plt.plot(idx[0])
# plt.plot(labels[0])
# plt.plot(truth[0])
# plt.show()

# update the timestep
timestep = len(x_train[0])
x_train = np.array(x_train)
label_train = np.array(label_train)
truth = np.array(truth)

'''build neural'''

input = Input(shape=(timestep, input_dim))
commons = input


'''common layers (encoder)'''
'''ConvNN before putting into LSTM'''
if input_dim > kernel_size:
    commons = Reshape(target_shape=(timestep, input_dim, 1))(commons)
    commons = TimeDistributed(Conv1D(16, kernel_size=kernel_size, data_format='channels_last', activation='relu'))(commons)
    commons = TimeDistributed(Conv1D(32, kernel_size=kernel_size, data_format='channels_last', activation='relu'))(commons)
    commons = TimeDistributed(Flatten(data_format='channels_last'))(commons)

'''bidirectional LSTM'''
o1 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(commons)
o1 = Dropout(0.2)(o1)
o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
o2 = Add()([o1, o2])
o2 = Dropout(0.2)(o2)

o3 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o2)
o3 = Add()([o1, o2, o3])
o3 = Dropout(0.2)(o3)

'''attention model for classifier'''
o4 = TimeDistributed(Dense(filter_size*2, activation='relu'))(o3)
o4 = TimeDistributed(Dense(filter_size*2, activation='softmax'))(o4)
o5 = Multiply()([o3, o4])

classifier = o5
classifier = TimeDistributed(Dense(filter_size*2, activation='relu'))(classifier)
classifier = TimeDistributed(Dense(6, activation='softmax'))(classifier)
classifier_model = Model(input, classifier)

'''denoiser (decode)'''
dae = o3
o6 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(dae)
o6 = Dropout(0.2)(o6)
o7 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o6)
o7 = Add()([o6, o7])
o7 = Dropout(0.2)(o7)

o8 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o7)
o8 = Add()([o6, o7, o8])
o8 = Dropout(0.2)(o8)

'''attention model for dae'''
o9 = TimeDistributed(Dense(filter_size*2, activation='relu'))(o8)
o9 = TimeDistributed(Dense(filter_size*2, activation='softmax'))(o9)
o10 = Multiply()([o8, o9])

dae = o10
dae = TimeDistributed(Dense(filter_size*2, activation='relu'))(dae)
dae = TimeDistributed(Dense(1, activation='linear'))(dae)

dae_model = Model(input, dae)

print(dae_model.summary())
print(classifier_model.summary())

dae_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), metrics=['mae'], loss='logcosh')

dae.trainable = False
classifier_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), metrics=['accuracy', 'categorical_accuracy'], loss='categorical_crossentropy')

hist_dae = dae_model.fit(x_train[:300], truth[:300], validation_data=(x_train[300:400], truth[300:400]), batch_size=batch_size, epochs=epochs, verbose=1)
hist_classifier = classifier_model.fit(x_train[:300], label_train[:300], validation_data=(x_train[300:400], label_train[300:400]), batch_size=batch_size, epochs=epochs, verbose=1)

idx_ref = idx[400:]
tested = x_train[400:]
sig_truth = truth[400:]
expected_label = label_train[400:]
sig_predicted = dae_model.predict(tested)
label_predicted = classifier_model.predict(tested)

# '''save the results'''
date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
dae_model.save('multitask_dae_ecgsim' + date_str + '.h5')
classifier_model.save('multitask_classifier_ecgsim'+date_str+'.h5')
hist_name = 'multitask_ecgsim_hist_' + date_str +'.dat'
pickle.dump((hist_dae, hist_classifier), open(hist_name, 'wb'))


plot_idx = 30

pr = [np.argmax(p) for p in label_predicted[plot_idx]]
ex = [np.argmax(p) for p in expected_label[plot_idx]]
plt.plot(pr)
plt.plot(ex)
plt.plot(tested[plot_idx])
plt.plot(sig_truth[plot_idx])
plt.plot(sig_predicted[plot_idx])
plt.plot(idx_ref[plot_idx])