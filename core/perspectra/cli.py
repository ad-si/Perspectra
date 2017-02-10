import os.path as path
import argparse
import textwrap

from .file_utils import renumber_pages
from .transformer import transform_image


def execute_arguments (arguments):
    parser = argparse.ArgumentParser(prog = 'perspectra')
    parser.add_argument(
        '--debug',
        help = 'Render debugging view',
        action = 'store_true',
    )

    # Add subparsers
    subparsers = parser.add_subparsers(
        title = 'subcommands',
        description = 'subcommands to handle files and correct photos',
        help = 'additional help',
        dest = 'subparser_name',
    )

    # Add subcommand 'correct'
    parser_correct = subparsers.add_parser(
        'correct',
        help = '''
            Pespectively correct and crop photos of documents.
        '''
    )
    parser_correct.add_argument(
        '--gray',
        help = 'Safe image as grayscale image',
        action = 'store_true',
        dest = 'output_in_gray',
    )
    parser_correct.add_argument(
        '--binary',
        help = 'Safe image as binary image',
        dest = 'binarization_method',
    )
    parser_correct.add_argument(
        '--marked-image',
        help = 'Copy of original image with marked corners',
        dest = 'marked_image_path',
    )
    parser_correct.add_argument(
        '--output',
        metavar = 'image-path',
        help = 'Output path of fixed image',
        dest = 'output_image_path',
    )
    parser_correct.add_argument(
        'input_image_path',
        nargs = '?',
        metavar = 'image-path',
        help='Path to image which shall be fixed',
    )
    parser_correct.set_defaults(func = transform_image)

    # Add subcommand 'renumber-pages'
    parser_rename = subparsers.add_parser(
        'renumber-pages',
        help = '''
            Renames the images in a directory according to their page numbers.
            The assumend layout is `cover -> odd pages -> even pages reversed`
        ''',
    )
    parser_rename.add_argument(
        'book_directory',
        metavar = 'book-directory',
        help = 'Path to directory containing the images of the pages',
    )
    parser_rename.set_defaults(func = renumber_pages)

    args = parser.parse_args(args = arguments)

    if not args.subparser_name:
        parser.print_help()
        return

    if args.input_image_path:
        args.input_image_path = path.abspath(args.input_image_path)

    if args.marked_image_path:
        args.marked_image_path = path.abspath(args.marked_image_path)

    if args.output_image_path:
        args.output_image_path = path.abspath(args.output_image_path)

    args.func(**vars(args))
