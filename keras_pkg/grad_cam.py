# -*- coding: utf-8 -*-

import os

from keras.preprocessing import image
import keras.models as models
import numpy as np


class Model:
    def __init__(self, weight_path, architecture_path=None):
        if architecture_path is None:
            self.model = models.load_model(weight_path)
        else:
            self.model = self.__load_architecture(architecture_path)

            self.model.load_weights(weight_path)

    def __load_architecture(self, path):
        with open(path) as file:
            model_str = file.read()
        # for json file
        if os.path.splitext(path)[-1] == '.json':
            return models.model_from_json(model_str)
        # for yml file
        else:
            return models.model_from_yaml(model_str)

    def __load_image(self, path, scaling_value):
        img_size = self.model.input_shape[1:-1]
        img = image.load_img(path, target_size=img_size)
        img_arr = image.img_to_array(img) / scaling_value
        return np.expand_dims(img_arr, axis=0)

    def grad_cam(self, image_path, scaling_value=255.0):
        target_image = self.__load_image(image_path, scaling_value)
        prediction_result = self.model.predict(target_image)
        predicted_class = np.argmax(prediction_result)
