import os.path as path
import argparse

def execute_arguments(arguments):
    from perspectra import file_utils
    parser = argparse.ArgumentParser(prog="perspectra")
    parser.add_argument(
        "--debug",
        help="Render debugging view",
        action="store_true",
    )

    # Add subparsers
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="subcommands to handle files and correct photos",
        help="additional help",
        dest="subparser_name",
    )

    # Add subcommand 'binarize'
    parser_binarize = subparsers.add_parser(
        "binarize",
        help="""
            Binarize image
        """,
    )
    parser_binarize.add_argument(
        "input_image_path",
        nargs="?",
        metavar="image-path",
        help="Path to image which shall be fixed",
    )
    parser_binarize.add_argument(
        "--method",
        help="Save image as binary image",
        dest="binarization_method",
    )
    parser_binarize.add_argument(
        "--no-clear-border",
        help="Do not remove any objects which touch the border",
        action="store_true",
        dest="shall_not_clear_border",
    )
    def binarize_handler(**kwargs):
        from perspectra import binarize
        binarize.binarize_image(**kwargs)

    parser_binarize.set_defaults(func=binarize_handler)

    # Add subcommand 'correct'
    parser_correct = subparsers.add_parser(
        "correct",
        help="""
            Pespectively correct and crop photos of documents.
        """,
    )
    parser_correct.add_argument(
        "--gray",
        help="Save image as grayscale image",
        action="store_true",
        dest="output_in_gray",
    )
    parser_correct.add_argument(
        "--binary",
        help="Save image as binary image",
        dest="binarization_method",
    )
    parser_correct.add_argument(
        "--no-clear-border",
        help="Do not remove any objects which touch the border",
        action="store_true",
        dest="shall_not_clear_border",
    )
    parser_correct.add_argument(
        "--marked-image",
        help="Copy of original image with marked corners",
        dest="image_marked_path",
    )
    parser_correct.add_argument(
        "--output",
        metavar="image-path",
        help="Output path of fixed image",
        dest="output_image_path",
    )
    parser_correct.add_argument(
        "input_image_path",
        nargs="?",
        metavar="image-path",
        help="Path to image which shall be fixed",
    )
    def transform_handler(**kwargs):
        from perspectra import transformer
        transformer.transform_image(**kwargs)

    parser_correct.set_defaults(func=transform_handler)

    # Add subcommand 'corners'
    parser_corners = subparsers.add_parser(
        "corners",
        help="""
            Returns the corners of the document in the image as
            [top-left, top-right, bottom-right, bottom-left]
        """,
    )
    parser_corners.add_argument(
        "input_image_path",
        nargs="?",
        metavar="image-path",
        help="Path to image to find corners in",
    )
    def corners_handler(**kwargs):
        from perspectra import transformer
        transformer.print_corners(**kwargs)

    parser_corners.set_defaults(func=corners_handler)

    # Add subcommand 'renumber-pages'
    parser_rename = subparsers.add_parser(
        "renumber-pages",
        help="""
            Renames the images in a directory according to their page numbers.
            The assumend layout is `cover -> odd pages -> even pages reversed`
        """,
    )
    parser_rename.add_argument(
        "book_directory",
        metavar="book-directory",
        help="Path to directory containing the images of the pages",
    )
    def rename_handler(**kwargs):
        from perspectra import file_utils
        file_utils.renumber_pages(**kwargs)

    parser_rename.set_defaults(func=rename_handler)

    args = parser.parse_args(args=arguments)

    if not args.subparser_name:
        parser.print_help()
        return

    if args.subparser_name == "binarize":
        if args.input_image_path:
            args.input_image_path = path.abspath(args.input_image_path)

    elif args.subparser_name == "corners":
        if args.input_image_path:
            args.input_image_path = path.abspath(args.input_image_path)

    else:
        if args.input_image_path:
            args.input_image_path = path.abspath(args.input_image_path)

        if args.image_marked_path:
            args.image_marked_path = path.abspath(args.image_marked_path)

        if args.output_image_path:
            args.output_image_path = path.abspath(args.output_image_path)

    args.func(**vars(args))
