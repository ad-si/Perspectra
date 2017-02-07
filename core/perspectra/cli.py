#! /usr/bin/env python3

import os.path as path
import argparse

from .transformer import transform_image


def execute_arguments (arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        help='Render debugging view',
        action='store_true',
    )
    parser.add_argument(
        '--gray',
        help='Safe image as grayscale image',
        action='store_true',
        dest='output_in_gray',
    )
    parser.add_argument(
        '--binary',
        help='Safe image as binary image',
        dest='binarization_method',
    )
    parser.add_argument(
        '--marked-image',
        help='Copy of original image with marked corners',
        dest='marked_image_path',
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

    if args.marked_image_path:
        args.marked_image_path = path.abspath(args.marked_image_path)

    if args.output_image_path:
        args.output_image_path = path.abspath(args.output_image_path)

    transform_image(**vars(args))
