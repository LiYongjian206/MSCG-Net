# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K
from keras.models import Model
from keras.layers.core import Lambda

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def resize_1d(array, shape):
    res = np.zeros(shape)
    if array.shape[0] >= shape:
        ratio = array.shape[0]/shape
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        res = res[::-1]
        array = array[::-1]
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        res = res[::-1]/(2*ratio)
        array = array[::-1]
    else:
        ratio = shape/array.shape[0]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = res[::-1]
        array = array[::-1]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = res[::-1]/2
        array = array[::-1]
    return res

def grad_cam(input_model, data, category_index, layer_name, nb_classes):
    def target_layer(x): return target_category_loss(
        x, category_index, nb_classes)
    x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape, name='cam')(x)
    model = Model(input_model.layers[0].input, x)
    loss = K.sum(model.layers[-1].output)

    conv_output = input_model.get_layer(layer_name).output

    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function(
        [model.input], [conv_output[0], grads])
    output, grads_val = gradient_function([data])
    output, grads_val = output[0, :], grads_val[0, :, :]
    weights = np.mean(grads_val, axis=(0))
    # print(weights.shape)
    # print(output.shape)
    cam = np.ones(output.shape[0: 1], dtype=np.float32)
    for i, w in enumerate(weights):
        # print(w * output[i])
        cam[i] = w * output[i]
    # print(cam)
    cam = resize_1d(cam, (data.shape[1]))
    cam = np.maximum(cam, 0)
    print(cam.shape)
    print(cam)
    heatmap = (cam - np.min(cam))/(np.max(cam) - np.min(cam)+1e-10)
    return heatmap



