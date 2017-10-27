from __future__ import print_function
from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Dense, Convolution2D

def svd_orthonormal(shape):
    print(shape)
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    print(shape)
    flat_shape = (shape[0], np.prod(shape[1:]))
    print(flat_shape)
    a = np.random.standard_normal(flat_shape)
    # a = np.random.normal(0, 0.01, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices = False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q

# do forward pass with a mini-batch
def forward_prop_layer(model, layer, batch_xs):
    intermediate_layer_model = Model(input=model.get_input_at(0), output=layer.get_output_at(0))
    activations = intermediate_layer_model.predict(batch_xs)
    return activations

def LSUV_init(model, batch_xs, layers_to_init = (Dense, Convolution2D), seed = 20):
    np.random.seed(seed)
    margin = 1e-6
    max_iter = 10
    layers_cnt = 0
    for layer in model.layers:
        if not any([type(layer) is class_name for class_name in layers_to_init]):
            continue
        print("cur layer is: ", layer.name)

        # as layers with few weights tend to have a zero variance, only do LSUV for complicated layers
        if np.prod(layer.get_output_shape_at(0)[1:]) < 32:
            print(layer.name, 'with output shape fewer than 32, not inited with LSUV')
            continue

        print('LSUV initializing', layer.name)
        layers_cnt += 1
        weights_all = layer.get_weights();
        weights = np.array(weights_all[0])

        # pre-initialize with orthonormal matrices
        weights = svd_orthonormal(weights.shape)
        biases = np.array(weights_all[1])
        weights_all_new = [weights, biases]
        layer.set_weights(weights_all_new)

        iter = 0
        target_var = 1.0 # the targeted variance

        layer_output = forward_prop_layer(model, layer, batch_xs)
        var = np.var(layer_output)
        print("cur var is: ", var)

        while (abs(target_var - var) > margin):
            # update weights based on the variance of the output
            weights_all = layer.get_weights()
            weights = np.array(weights_all[0])
            # print(weights)
            biases = np.array(weights_all[1])
            if np.abs(np.sqrt(var)) < 1e-7: break  # avoid zero division

            weights = weights / np.sqrt(var) # try to scale the variance to the target
            print(weights)
            weights_all_new = [weights, biases]
            layer.set_weights(weights_all_new) # update new weights

            layer_output = forward_prop_layer(model, layer, batch_xs)
            var = np.var(layer_output)
            print("cur var is: ", var)

            iter = iter + 1
            if iter > max_iter:
                break

    print('LSUV: total layers initialized', layers_cnt)
    return model