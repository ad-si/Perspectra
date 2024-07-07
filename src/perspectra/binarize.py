import os
import base64
import imageio
import logging
import numpy

import skimage
from skimage import (
    morphology,
    segmentation,
    util,
)
from skimage.color import rgb2gray
from skimage.filters import (
    rank,
    gaussian,
    threshold_local,
    threshold_otsu,
    threshold_sauvola,
)
from skimage.util import img_as_ubyte
from perspectra import multipass_cleaner


class ImageDebugger:
    def __init__(self, level, base_path):
        self.level = level
        self.base_path = base_path
        self.step_counter = 0

    def set_level(self, level):
        self.level = level
        return self

    def set_base_path(self, base_path):
        self.base_path = base_path
        return self

    def save(self, name, image):
        if self.level != "debug":
            return
        self.step_counter += 1
        imageio.imwrite(
            os.path.join(self.base_path, f"{self.step_counter}-{name}.png"),
            image,
        )
        return self


def clear(binary_image, debugger):
    inverted_image = util.invert(binary_image)
    inverted_cleared_image = segmentation.clear_border(inverted_image)
    cleared_image = util.invert(inverted_cleared_image)
    debugger.save("cleared_image", cleared_image)
    return cleared_image


def denoise(binary_image, debugger):
    inverted_image = util.invert(binary_image)
    inverted_denoised_image = multipass_cleaner.remove_noise(inverted_image)
    denoised_image = util.invert(inverted_denoised_image)
    debugger.save("denoised_image", denoised_image)

    return denoised_image


def erode(image, image_name, debugger):
    eroded_image = morphology.erosion(
        util.img_as_ubyte(image),
        morphology.disk(25),
    )
    debugger.save(f"eroded_{image_name}", eroded_image)
    return eroded_image


def binarize(image, debugger, method="sauvola"):
    radius = 3

    gray_image = rgb2gray(image)
    debugger.save("gray_image", gray_image)

    if method == "sauvola":
        window_size = 3  # Minimal window size
        window_size += gray_image.size // (2**20)  # Relative to image size
        window_size += 1 if (window_size % 2 == 0) else 0  # Must always be odd
        logging.info(f"window_size: {window_size}")

        thresh_sauvola = numpy.nan_to_num(
            threshold_sauvola(
                image=gray_image,
                window_size=window_size,
                k=0.3,  # Attained through experimentation
            )
        )
        debugger.save("thresh_sauvola", thresh_sauvola)
        binarized_image = gray_image > thresh_sauvola

    elif method == "local":
        binarized_image = gray_image > threshold_local(
            image=gray_image,
            block_size=radius,
        )

    elif method == "niblack":
        sigma = gray_image.size // (2**17)
        thresh_niblack = skimage.filters.threshold_niblack(
            image=gray_image,
            window_size=radius,
            k=0.08,
        )
        binarized_image = gray_image > thresh_niblack

    elif method == "gauss-diff":
        sigma = gray_image.size // (2**17)
        high_frequencies = gray_image - gaussian(
            image=gray_image,
            sigma=sigma,
        )
        thresh = threshold_otsu(high_frequencies)
        binarized_image = high_frequencies > thresh

    elif method == "local-otsu":
        warped_image_ubyte = img_as_ubyte(gray_image)
        selem = morphology.disk(radius)
        local_otsu = rank.otsu(warped_image_ubyte, selem)
        binarized_image = warped_image_ubyte >= local_otsu

    else:
        raise TypeError(f"{method} is no supported binarization method")

    debugger.save("binarized_image", binarized_image)

    return binarized_image


def get_binarized_image(
    input_image_path,
    binarization_method,
    shall_clear_border,
    debugger,
):
    image = imageio.imread(input_image_path, rotate=True)

    binarized_image = binarize(
        image=image,
        method=binarization_method,
        debugger=debugger,
    )
    if shall_clear_border:
        cleared_image = clear(binarized_image, debugger)
        erode(cleared_image, "cleared", debugger)
        denoised_image = denoise(cleared_image, debugger)
    else:
        erode(binarized_image, "binarized", debugger)
        denoised_image = denoise(binarized_image, debugger)

    erode(denoised_image, "denoised", debugger)

    return denoised_image


def binarize_image(**kwargs):
    binarization_method = kwargs.get("binarization_method")
    shall_clear_border = not kwargs.get("shall_not_clear_border", False)
    input_image_path = kwargs.get("input_image_path")
    debug = kwargs.get("debug", False)

    file_name_segments = os.path.splitext(os.path.basename(input_image_path))
    basename = file_name_segments[0]
    random_string = (
        base64.b64encode(os.urandom(3))
        .decode("utf-8")
        .replace("+", "-")
        .replace("/", "_")
    )
    output_base_path = os.path.join(os.path.dirname(input_image_path), basename)

    output_image_path = (
        kwargs.get("output_image_path")
        or f"{output_base_path}-fixed_{random_string}.png"
    )

    if not input_image_path:
        raise FileNotFoundError(
            f"An input image and not {input_image_path} must be specified"
        )

    debugger = ImageDebugger(
        level="debug" if debug else "",
        base_path=output_base_path,
    )

    binarized_image = get_binarized_image(
        input_image_path, binarization_method, shall_clear_border, debugger
    )

    if not debug:
        imageio.imwrite(output_image_path, binarized_image)
