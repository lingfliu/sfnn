from multiprocessing import Pool, Manager
from wfdb_tool import prepare_training_set
from keras.utils import to_categorical

'''lib loading error prevention'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''scientific packages'''
import numpy as np
import matplotlib.pyplot as plt
import pickle


'''global parameters'''
input_dim = 5000
output_dim = 5000

def label_expand(label_pos, label, queue):
    lb_new = np.zeros(np.shape(label))
    idx = 0
    while idx < len(label):
        if label[idx] == 1:
            for idx2 in range(idx-40, idx+40, 1):
                if len(label)-1 >= idx2 >= 0:
                    lb_new[idx2] = 1
        idx += 1
    print('expanding label at ', label_pos)
    queue.put((label_pos, lb_new))


def x_train_arrange(pos, x, queue):
    seq = np.array([x[i*stride:i*stride+input_dim] for i in range((len(x)-input_dim)//stride)])
    print('arrange x_train at ', pos)
    queue.put((pos, seq))


def y_train_arrange(pos, y, queue):
    seq = np.array([y[i*stride+input_dim//2-output_dim//2:i*stride+input_dim//2 - output_dim//2 + output_dim] for i in range( (len(y)-input_dim)//stride )])
    seq = to_categorical(seq, num_classes=2)
    print('arrange y_train at ', pos)
    queue.put((pos, seq))

if __name__ == '__main__':
    # fs = 360  # sampling rate of mitdb
    # '''load training data and shuffle'''
    # (input, label_raw, label_typ_raw) = prepare_training_set(set_len=5000)
    # paired = []
    # for idx in range(len(input)):
    #     paired.append((input[idx], label_raw[idx], label_typ_raw[idx]))
    # np.random.shuffle(paired)
    # input = []
    # label_raw = []
    # for (i, l, t) in paired:
    #     input.append(i)
    #     label_raw.append(l)
    #     label_typ_raw.append(t)
    #
    # '''save tmp data'''
    # pickle.dump((input, label_raw, label_typ_raw), open('mitdb_500_qrs.dat', 'wb'))

    '''load tmp data'''
    (input, label_raw, label_typ_raw) = pickle.load(open('mitdb_500_qrs.dat', 'rb'))
    print('finish loading raw data')

    '''label expansion'''
    label = []
    pool = Pool(processes=40)
    queue = Manager().Queue(maxsize=20000)
    cnt = 0
    for i in range(len(label_raw)):
        pool.apply_async(label_expand, args=(i, label_raw[i], queue, ))
    pool.close()
    pool.join()

    label_ex = []
    while True:
        try:
            (i, lb) = queue.get_nowait()
            label_ex.append((i, lb))
        except:
            break
    print('finish label expansion ', len(label_ex))
    label_ex = sorted(label_ex, key=lambda x:x[0])
    label = []
    for (i, lb) in label_ex:
        label.append(lb)

    '''test code'''
    # plt.plot(input[7])
    # plt.plot(label[7])
    # plt.show()

    x_train = []
    for dat in input:
        dat = np.reshape(dat, [len(dat), 1])
        x_train.append(dat)

    y_train = []
    for lb in label:
        lb_1 = np.reshape(lb, [len(lb), 1])
        lb_1 = to_categorical(lb_1, num_classes=2)
        y_train.append(lb_1)

    pickle.dump((label, x_train, y_train, input_dim, output_dim), open('mitdb_500_train_fcn.dat', 'wb'))
