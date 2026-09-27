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
        help="Binarization method to use",
        choices=[
            "gauss-diff",
            "local-otsu",
            "local",
            "niblack",
            "sauvola",
        ],
        default="sauvola",
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
        choices=[
            "gauss-diff",
            "local-otsu",
            "local",
            "niblack",
            "sauvola",
        ],
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
            The assumed layout is `cover -> odd pages -> even pages reversed`
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

    # Add subcommand 'extract-pages'
    parser_extract = subparsers.add_parser(
        "extract-pages",
        help="""
            Extract a photo of each page from a video of a book being
            flipped through. A page is captured whenever a short clicking
            sound (e.g. a tongue pop) is made while the page is held in focus.
        """,
    )
    parser_extract.add_argument(
        "input_video_path",
        metavar="video-path",
        help="Path to the video",
    )
    parser_extract.add_argument(
        "--output",
        metavar="directory",
        help="Directory to save the pages in (default: <video-name>_pages)",
        dest="output_directory",
    )
    parser_extract.add_argument(
        "--frame-window",
        type=float,
        default=0.2,
        metavar="seconds",
        help="""
            Use the sharpest frame within this time window around the click.
            0 uses the frame at the click. (default: %(default)s)
        """,
    )
    parser_extract.add_argument(
        "--min-contrast",
        type=float,
        default=28.0,
        metavar="dB",
        help="""
            Minimum level of a click above the signal in the 250 ms before it.
            Not required if there is only background noise before it.
            (default: %(default)s)
        """,
    )
    parser_extract.add_argument(
        "--min-decay",
        type=float,
        default=12.0,
        metavar="dB",
        help="""
            Minimum level decrease 30 - 80 ms after the onset of a click,
            relative to the background noise (default: %(default)s)
        """,
    )
    parser_extract.add_argument(
        "--min-snr",
        type=float,
        default=15.0,
        metavar="dB",
        help="""
            Minimum signal-to-noise ratio of a click
            relative to the background noise (default: %(default)s)
        """,
    )
    parser_extract.add_argument(
        "--max-level-spread",
        type=float,
        default=15.0,
        metavar="dB",
        help="""
            Discard sounds which are more than this much quieter
            than the median click (default: %(default)s)
        """,
    )
    parser_extract.add_argument(
        "--min-gap",
        type=float,
        default=0.5,
        metavar="seconds",
        help="Minimum time between two clicks (default: %(default)s)",
    )
    def extract_handler(**kwargs):
        from perspectra import page_extractor
        page_extractor.extract_pages(**kwargs)

    parser_extract.set_defaults(func=extract_handler)

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

    elif args.subparser_name in ("extract-pages", "renumber-pages"):
        pass

    else:
        if args.input_image_path:
            args.input_image_path = path.abspath(args.input_image_path)

        if args.image_marked_path:
            args.image_marked_path = path.abspath(args.image_marked_path)

        if args.output_image_path:
            args.output_image_path = path.abspath(args.output_image_path)

    args.func(**vars(args))
