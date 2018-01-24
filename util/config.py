# -*- coding: utf-8 -*-

import yaml

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
        self.layer = model_dict['layer']
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
        self.output = image_dict['output']
        self.source = None
        if 'source' in image_dict.keys():
            self.source = Source(image_dict['source'])

class Source:
    def __init__(self, source_dict):
        self.path = source_dict['path']
        self.definition = source_dict['definition']
        self.args = []
        if 'args' in source_dict.keys():
            self.args = source_dict['args']