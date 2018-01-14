# -*- coding: utf-8 -*-

import argparse
import keras_pkg.grad_cam as k_grad_cam


def cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('image',
                        type=str,
                        help='set image path')
    parser.add_argument('-w', '--weight',
                        required=True,
                        help='set weight file (h5) path')
    parser.add_argument('-m', '--model',
                        help='set architecture file (json or yml) path')

    return parser.parse_args()


def main():
    # command line parse
    args = cmd()

    # create target model for grad-cam
    if args.model is None:
        target = k_grad_cam.Model(args.weight)
    else:
        target = k_grad_cam.Model(args.weight, args.model)

    # show model summary
    target.model.summary()

    # grad-cam
    target.grad_cam(args.image)


if __name__ == '__main__':
    main()
