from sig_simulator import simu_sinusoid_sig, simu_gaussian
from matplotlib import pyplot as pp
import math

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras import backend as K

def main():
    #sps = 200
    #sig_len = 6000
    #(x,y) = simu_sinusoid_sig(sps=sps, x_len=sig_len, phase=math.pi/6)
    #n = simu_gaussian(x, 0, 0.1)


    #pp.plot(x,y+n)
    #pp.show()
    
    model = Sequential()
    model.add(Dense(units=128, activation='selu', input_dim=100))

    row=28
    col=28
    batch_size = 64
    class_num = 10
    epochs = 200
    
    (x_train,y_train),(x_test,y_test) = mnist.load_data()

    if K.image_data_format() == 'channel_first':
        x_train = x_train.reshape(x_train.shape[0], 1, row, col)
        x_test = x_test.reshape(x_test.shape[0], 1, row, col)
        input_shape = (1, row, col)
    else:
        x_train = x_train.reshape(x_train.shape[0], row, col, 1)
        x_test = x_test.reshape(x_test.shape[0], row, col, 1)
        input_shape = (row, col, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, class_num)
    y_test = keras.utils.to_categorical(y_test, class_num)


    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(class_num,activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
                    metrics=['accuracy'])

    model.fit(x_train,y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test,y_test))

    score = model.evaluate(x_test, y_test, verbose=1)

if __name__ == '__main__':
    main()
