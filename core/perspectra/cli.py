#! /usr/bin/env python3

import os.path as path
import argparse

import transformer


def execute_arguments (arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        help='Render debugging view',
        action='store_true',
    )
    parser.add_argument(
        '--gray',
        help='Safe image in grayscale',
        action='store_true',
        dest='output_in_gray',
    )
    parser.add_argument(
        '--output',
        metavar='image-path',
        help='Output path of fixed image',
        dest='output_image_path',
    )
    parser.add_argument(
        'input_image_path',
        metavar='image-path',
        help='Path to image which shall be fixed',
    )

    args = parser.parse_args(args=arguments)

    if args.input_image_path:
        args.input_image_path = path.abspath(args.input_image_path)

    if args.output_image_path:
        args.output_image_path = path.abspath(args.output_image_path)

    transformer.transform_image(**vars(args))
