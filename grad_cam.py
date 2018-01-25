# -*- coding: utf-8 -*-

import argparse
import importlib.machinery as im_machinery
import os
import glob
import imghdr
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


def __get_file_path_from_dir(path, extension):
    return [f for f in glob.glob(path + '/*') if imghdr.what(f) in extension]
    

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

    # images check (file or dir) and get images path
    if os.path.isfile(config.image.path):
        image_paths = [config.image.path]
    else:
        image_paths = __get_file_path_from_dir(config.image.path, ['jpg', 'png'])

    # grad-cam
    if config.image.source is None:
        results = k_grad_cam.exec(
            model,
            config.model.layer,
            image_paths)
    else:
        preprocessing_func = __function_loader(
            config.image.source.path,
            config.image.source.definition)

        results = k_grad_cam.exec(
            model,
            config.model.layer,
            image_paths,
            preprocessing_func)
    
    return results, image_paths, model.name


def main():
    # command line parse
    args = __cmd()
    # get config
    config = Config(args.config_file)

    # for keras
    if config.framework == 'keras':
        results, image_paths, model_name = __keras_grad_cam(config)
    # Unimplemented
    else:
        print('Unimplemented ' + config.framework)
    
    # check output dir
    if not os.path.isdir(config.image.output):
        os.makedirs(config.image.output)
    # output file
    for (idx, (cam, heatmap)) in enumerate(results): 
        img_name, extension = __get_filename(image_paths[idx])
        target_name =  '%s-%s' % (model_name, config.model.layer)
        img_file_name = '%s%s' % (img_name, extension)
        output_name = 'gradcam-{0}-{1}-{2}'.format(target_name, img_file_name, extension)
        cv2.imwrite(os.path.join(config.image.output, output_name), cam)


if __name__ == '__main__':
    main()
