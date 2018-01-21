# -*- coding: utf-8 -*-

import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path) as f:
           config = yaml.load(f)
        self.framework = list(config.keys())[0]
        self.target = Target(config[self.framework]['target'])

class Target:
    def __init__(self, target_dict):
        self.params = target_dict['params']
        self.layer = target_dict['layer']
        self.image = target_dict['image']
        self.architecture = None
        if 'architecture' in target_dict.keys():
            self.architecture = target_dict['architecture']
        self.preprocessing = None
        if 'preprocessing' in target_dict.keys():
            self.preprocessing = target_dict['preprocessing']
