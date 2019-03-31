from wfdb_tool import prepare_training_set

import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import CuDNNLSTM, Add, Multiply, Bidirectional, Dense, LSTM
from keras.layers import Input, BatchNormalization
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

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                        inter_op_parallelism_threads=num_cores, \
                        allow_soft_placement=True, \
                        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True

session = tf.Session(config=config)
K.set_session(session)


'''scientific packages'''
import numpy as np
import pickle
import datetime

if __name__ == '__main__':
    fs = 500  # sampling rate of mitdb

    (input, label_raw, label_typ_raw) = pickle.load(open('mitdb_500_qrs.dat', 'rb'))
    (label, x_train, y_train, input_dim, output_dim, stride) = pickle.load(open('mitdb_500_train_lstm.dat', 'rb'))

    # nn hyper params
    batch_size = 40
    epochs = 100
    filter_size = 80
    kernel_size = 4
    dropout = 0.2

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
    classifier = TimeDistributed(Dense(2, activation='softmax'))(classifier)


    model = Model(input, classifier)
    print(model.summary())

    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), metrics=['accuracy', 'categorical_accuracy'], loss='categorical_crossentropy')

    '''65% for training, 20% for validation, 15% for testing'''
    len_set = np.shape(x_train)[0]
    len_train = (int)(len_set*0.65)
    len_valid = (int)(len_set*0.2)
    hist = model.fit(x_train[:len_train], y_train[:len_train], validation_data=(x_train[len_train:len_train+len_valid], y_train[len_train:len_train+len_valid]), batch_size=batch_size, epochs=epochs, verbose=1)


    tested = x_train[len_train+len_valid:]
    expected = y_train[len_train+len_valid:]
    predicted = model.predict(tested)

    # '''save the results'''
    date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model.save('mitdb_500_semantic_seg_' + date_str + '.h5')
    hist_name = 'mitdb_500_semantic_seg_' + date_str +'.hist'
    pickle.dump(hist, open(hist_name, 'wb'))

