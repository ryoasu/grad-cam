# -*- coding: utf-8 -*-

import yaml
import os
import glob
import imghdr


class Config:
    def __init__(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f)
        self.framework = list(config.keys())[0]
        self.model = Model(config[self.framework]['model'])
        self.image = Image(config[self.framework]['image'])


class Model:
    def __init__(self, model_dict):
        self.params = model_dict['params']
        self.layers = model_dict['layers']
        self.architecture = None
        if 'architecture' in model_dict.keys():
            self.architecture = model_dict['architecture']
        self.source = None
        if 'source' in model_dict.keys():
            self.source = Source(model_dict['source'])

        # check describe only one of architecture and source in config
        self.__check()

    def __check(self):
        message = 'Please describe only one of architecture and source in config'
        assert (self.architecture is None) or (self.source is None), message


class Image:
    def __init__(self, image_dict):
        self.path = image_dict['path']
        self.output = './'
        if 'output' in image_dict.keys():
            self.output = image_dict['output']
        self.source = None
        if 'source' in image_dict.keys():
            self.source = Source(image_dict['source'])
        
        # check image path (or images dir path)
        self.__check()

    def __check(self):
        message = 'Please set image path or images dir path in config'
        if os.path.isdir(self.path):
            check_list = [imghdr.what(f) for f in glob.glob(self.path + '/*')]
            is_image = (None not in check_list) and (check_list != [])
        else:
            is_image = imghdr.what(self.path) is not None
        assert is_image, message


class Source:
    def __init__(self, source_dict):
        self.path = source_dict['path']
        self.definition = source_dict['definition']
        self.args = []
        if 'args' in source_dict.keys():
            self.args = source_dict['args']