# -*- coding: utf-8 -*-

import os

import numpy as np
from keras import models
from keras.engine.training import Model
from keras.preprocessing import image


def load_image(path, size, preprocess=False):
    img = image.load_img(path, target_size=size)
    img_arr = image.img_to_array(img)
    if preprocess:
        img_arr = img_arr / 255.
    return img_arr


def image_to_array(model, path, size, preprocessing=None):
    if preprocessing is None:
        preprocessed_image = load_image(path, size, preprocess=True)
    else:
        preprocessed_image = preprocessing(path, size)
    
    return np.expand_dims(preprocessed_image, axis=0)


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


def show_predicted_class(model, image_paths, preprocessing=None):
    for image_path in image_paths:
        image_size = model.input_shape[1:-1]
        input_image = image_to_array(model, image_path, image_size, preprocessing)
        predicted_class = model.predict(input_image)
        
        # show predicted class 
        print_str = "image: {0}, class: {1}".format(
            os.path.basename(image_path), 
            np.argmax(predicted_class))
        print(print_str)
