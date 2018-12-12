import numpy as np

def init_layer(input_len, output_len, layer_param, type='linear'):
    param = np.matrix([1]*input_len, [1]*output_len)
    layer_param.append(param)
    return layer_param

def neuron(w, x, type='linear'):
    inner_product = np.dot(w[1:], x) + w[0]
    if type == 'linear':
        return inner_product
    if type == 'sigmoid':
        return inner_product * (1 - inner_product) if derivative else 1 / (1 + np.exp(-x))


'''back propagation'''
def bp():
    pass

'''convolution'''
def convol(input, pooling='median'):
    pass






