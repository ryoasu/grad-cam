# -*- coding: utf-8 -*-

import os
import types

from tqdm import tqdm
from PIL import Image
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.engine.training import Model
import keras.backend as K
import keras.models as models
import numpy as np
import tensorflow as tf
import cv2

# set test phase
K.set_learning_phase(0)


def __load_image(path, size, preprocess=False):
    img = image.load_img(path, target_size=size)
    img_arr = image.img_to_array(img)
    if preprocess:
        img_arr = img_arr / 255.
    return img_arr


def get_model(weight_path, architecture=None):
    def load_architecture(architecture):
        with open(architecture) as file:
            model_str = file.read()
        # for json file
        if os.path.splitext(architecture)[-1] == '.json':
            return models.model_from_json(model_str)
        # for yml file
        else:
            return models.model_from_yaml(model_str)    
    
    if isinstance(architecture, Model):
        architecture.load_weights(weight_path)
        return architecture
    elif architecture is None:
        return models.load_model(weight_path)
    else:
        model = load_architecture(architecture)
        model.load_weights(weight_path)
        return model


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def normalize(x):
    # calculate RMS (root mean square)
    rms = K.sqrt(K.mean(K.square(x)))
    # normalize
    return x / (rms + 1e-5)


def relu(x):
    return np.maximum(x, 0)


def grad_cam(model, target_layer, image_path, preprocessing=None):
    # target image for grad-cam
    image_size = model.input_shape[1:-1]
    if preprocessing is None:
        preprocessed_image = __load_image(image_path, image_size, preprocess=True)
    else:
        preprocessed_image = preprocessing(image_path, image_size)
    input_image = np.expand_dims(preprocessed_image, axis=0)

    # get prediction result and
    prediction_result = model.predict(input_image)
    predicted_class = np.argmax(prediction_result)

    # Add layers to model for grad-cam (target_model)
    nb_classes = model.output_shape[-1]
    loss_layer = lambda x: target_category_loss(x, predicted_class, nb_classes)
    target_model = models.Model(
        model.input, 
        Lambda(loss_layer, output_shape=lambda x: x)(model.output))

    # implement function for calculate gradient (gradient_function)
    loss = K.sum(target_model.layers[-1].output)
    conv_output = target_model.get_layer(target_layer).output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([target_model.layers[0].input],[conv_output, grads])

    # calculate gradient and output
    output, grads_val = gradient_function([input_image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    # calculate weights of feature map
    weights = np.mean(grads_val, axis=(0, 1))

    # calculate heatmap
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    cam = cv2.resize(cam, (image_size[1], image_size[0]))
    cam = relu(cam)
    heatmap = cam / np.max(cam)

    # load original image (pixcel value: [0..255])
    origin_image = __load_image(image_path, image_size)

    # create cam
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(origin_image)
    cam = 255 * cam / np.max(cam)

    return np.uint8(cam), heatmap


def exec(model, target_layers, image_paths, preprocessing=None):
    results = []
    tqdm_target_layers = tqdm(target_layers)
    for target_layer in tqdm_target_layers:
        tqdm_target_layers.set_description('LAYERS: %s' % target_layer)
        
        layer_results = []
        tqdm_image_paths = tqdm(image_paths)
        for image_path in tqdm_image_paths:
            tqdm_image_paths.set_description('IMAGES: %s' % image_path)
            layer_results.append(
                grad_cam(model, target_layer, image_path, preprocessing))

        results.append(layer_results)

    return results
