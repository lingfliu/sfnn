from keras.models import Model, load_model

import pickle

model_h5 = 'dae_ecgsim.h5'
data = 'ecgsim.dat'

model = load_model(model_h5)
(sigs, sigs_noisy, param_sig, param_nn, x_train, y_train) = pickle.load(open(data, 'rb'))

for m in range(4):
    for n in range(4):
        idx = n*4+m
        if idx >= 50:
            break
        ax = plt.subplot2grid((4,4),(m,n))
        ax.plot(predicted[idx])
        ax.plot(sigs[idx+predicted][input_dim:])
        ax.plot(sigs_noisy[idx+predicted][input_dim:])
plt.show()
