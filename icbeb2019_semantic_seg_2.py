from wfdb_tool import prepare_training_set
from icbeb_tool import  load_icbeb2019

import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import CuDNNLSTM, Add, Multiply, Bidirectional, Dense, LSTM
from keras.layers import Reshape, Input, Flatten, BatchNormalization
from keras.layers import Dropout, TimeDistributed

import matplotlib.pyplot as plt

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


fs = 500 # sampling rate of mitdb

'''load tmp data'''
(data, label_raw) = pickle.load(open('icbeb_test.tmp', 'rb'))

'''temporal revision on the label'''
label_unique = []
for lb in label_raw:
    lb_new = np.zeros(np.shape(lb))
    idx = 0
    label_start = -1
    label_end = -1
    while idx < len(lb):
        if lb[idx] == 1:
            # mark as start
            if label_start < 0:
                label_start = idx
        else:
            if label_start >= 0:
                label_end = idx-1
                label_idx_central = (label_start+label_end)//2

                lb_new[label_idx_central] = 1
                label_start = -1
                label_end = -1
            else:
                pass
        idx += 1

    label_unique.append(lb_new)

'''split qrs into two segments'''
label = []
for lb in label_unique:
    lb_ex = np.zeros(np.shape(lb))
    idx = 0
    while idx < len(lb):
        if lb[idx] == 1:
            for idx2 in range(idx-80, idx, 1):
                if 0 <= idx2 <= len(lb)-1:
                    lb_ex[idx2] = 1

            for idx2 in range(idx, idx+80, 1):
                if 0 <= idx2 <= len(lb)-1:
                    lb_ex[idx2] = 2

        idx += 1

    label.append(lb_ex)

'''global parameters'''
input_dim = 1
output_dim = 1

if input_dim < output_dim:
    print('input_dim smaller than output_dim, quit task')

stride = output_dim
timestep = 0

# hyper params
batch_size = 40
epochs = 400
filter_size = 80
kernel_size = 4
dropout = 0.2


# stagging the signal
x_train = []
for dat in data:
    seq = np.array([dat[i*stride:i*stride+input_dim] for i in range((len(dat)-input_dim)//stride)])
    x_train.append(seq)

y_train = []
for lb in label:
    y = np.array([lb[i*stride+input_dim//2-output_dim//2:i*stride+input_dim//2 - output_dim//2 + output_dim] for i in range( (len(lb)-input_dim)//stride )])
    y = to_categorical(y, num_classes=3)
    y_train.append(y)

timestep = len(x_train[0])
x_train = np.array(x_train)
y_train = np.array(y_train)

'''build neural'''
input = Input(shape=(timestep, input_dim))
classifier = input

'''bidirectional LSTM'''
o1 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(classifier)
o1 = Dropout(0.2)(o1)
o1 = BatchNormalization()(o1)

o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
o2 = Add()([o1, o2])
o2 = Dropout(0.2)(o2)
o2 = BatchNormalization()(o2)

o3 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o2)
o3 = Add()([o1, o2, o3])
o3 = Dropout(0.2)(o3)
o3 = BatchNormalization()(o3)

'''attention model'''
oa = TimeDistributed(Dense(filter_size*2, activation='softmax'))(o3)
o3 = Multiply()([o3, oa])

classifier = o3
classifier = TimeDistributed(Dense(filter_size*2, activation='relu'))(classifier)
classifier = TimeDistributed(Dense(3, activation='softmax'))(classifier)


model = Model(input, classifier)
print(model.summary())

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), metrics=['accuracy', 'categorical_accuracy'], loss='categorical_crossentropy')

hist = model.fit(x_train[:1500], y_train[:1500], validation_data=(x_train[1500:1800], y_train[1500:1800]), batch_size=batch_size, epochs=epochs, verbose=1)


tested = x_train[1800:]
expected = y_train[1800:]
predicted = model.predict(tested)

# '''save the results'''
date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model.save('icbeb2019_semantic_seg_' + date_str + '.h5')
hist_name = 'icbeb2019_semantic_seg_' + date_str +'.hist'
pickle.dump(hist, open(hist_name, 'wb'))

