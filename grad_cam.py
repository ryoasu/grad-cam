# -*- coding: utf-8 -*-

import argparse
import importlib.machinery as im_machinery
import os
import types

import cv2
import keras_pkg.grad_cam as k_grad_cam
from util.config import Config


def __cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='set config file (.yml)')

    return parser.parse_args()


def __function_loader(mod_path, func_name, mod_name='mod'):
    loader = im_machinery.SourceFileLoader(mod_name, mod_path)
    mod = types.ModuleType(loader.name)
    # import module
    loader.exec_module(mod)
    # Check whether module is defined in function
    assert func_name in dir(mod), '{0} is undefined in {1}'.format(func_name, mod_path)
    func_str = mod_name + '.' + func_name

    return eval(func_str)


def __get_filename(path):
    return os.path.splitext(os.path.basename(path))


def __keras_grad_cam(config):
    # get model for grad-cam
    if (config.model.architecture is None) != (config.model.source is None):
        if config.model.architecture is None:
            # load model definition
            model_definition = __function_loader(
                config.model.source.path,
                config.model.source.definition)
            # get model from source code
            model = k_grad_cam.get_model(
                config.model.params,
                model_definition(*config.model.source.args))
        else:
            # get model from archarchitecture file
            model = k_grad_cam.get_model(
                config.model.params,
                config.model.architecture)
    else:
        # get model from params file
        model = k_grad_cam.get_model(config.model.params)
    
    # show model summary
    model.summary()

    # grad-cam
    if config.image.source is None:
        cam, heatmap = k_grad_cam.grad_cam(
            model,
            config.model.layer,
            config.image.path)
    else:
        preprocessing_func = __function_loader(
            config.image.source.path,
            config.image.source.definition)

        cam, heatmap = k_grad_cam.grad_cam(
            model,
            config.model.layer,
            config.image.path,
            preprocessing_func)

    # output file
    img_name, extension = __get_filename(config.image.path)
    output_name = 'gradcam-{0}-{1}{2}'.format(model.name, img_name, extension)
    cv2.imwrite(output_name, cam)


def main():
    # command line parse
    args = __cmd()
    # get config
    conf = Config(args.config_file)

    # for keras
    if conf.framework == 'keras':
        __keras_grad_cam(conf)
    # Unimplemented
    else:
        print('Unimplemented ' + conf.framework)


if __name__ == '__main__':
    main()
