import json
import os
import base64
import logging
from typing import Tuple

import imageio.v3 as imageio
import numpy
import numpy as np
import skimage
from skimage import (
    draw,
    exposure,
    feature,
    io,
    morphology,
    segmentation,
    transform,
    util,
)
from skimage.color import rgb2gray, label2rgb
from skimage.feature import (
    corner_peaks,
    corner_foerstner,
)
from skimage.filters import (
    sobel,
    gaussian,
    threshold_otsu,
    threshold_sauvola,
)
from skimage.segmentation import watershed
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
        image_path = os.path.join(
            self.base_path,
            f"{self.step_counter}-{name}.png",
        )
        imageio.imwrite(image_path, image)
        logging.info(f"Stored image: {image_path}")
        return self


def load_image(file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    return io.imread(file_path)


def get_img_corners(shape):
    rows = shape[0]
    colums = shape[1]
    return [
        (0, 0),
        (0, colums - 1),
        (rows - 1, colums - 1),
        (rows - 1, 0),
    ]


def cartesian_to_polar(points):
    """
    >>> cartesian_to_polar(np.array([[-1,1], [-1,-1], [1,-1], [1,1]]))
    array([[  1.41421356,  45.        ],
           [  1.41421356, 135.        ],
           [  1.41421356, 225.        ],
           [  1.41421356, 315.        ]])
    """
    x, y = points[:, 1], points[:, 0]
    r = np.hypot(x, y)
    thetas = np.arctan2(y, x)

    def norm_theta(theta):
        if theta < 0:
            return -theta
        else:
            return -(theta - 360)

    v_norm_theta = np.vectorize(norm_theta)

    thetas_norm = v_norm_theta(np.degrees(thetas))

    polar_points = np.column_stack((r, thetas_norm))

    return polar_points


def get_sorted_corners(img_size, corners):
    """
    Corners sorted from upper left corner (smallest row, column) clockwise
    using the angle of polar coordinates.

    >>> get_sorted_corners((0, 0), np.array([[-1,-1], [-1,1], [1,-1], [1,1]]))
    array([[-1, -1],
           [-1,  1],
           [ 1,  1],
           [ 1, -1]])

    >>> get_sorted_corners((250, 200), np.array([
    ...     [ 14,  46],
    ...     [234, 144],
    ...     [ 14, 140],
    ...     [234,  47],
    ...     [ 44,  30],
    ... ]))
    array([[ 14,  46],
           [ 14, 140],
           [234, 144],
           [234,  47],
           [ 44,  30]])
    """
    # Shift coordinate sytem
    # TODO: Find centroid of corners instead of using image center
    rowOffset = img_size[0] / 2
    colOffset = img_size[1] / 2

    moved_corner_points = corners - np.array([rowOffset, colOffset])

    polar_points = cartesian_to_polar(moved_corner_points)

    indices = np.argsort(polar_points[:, 1])
    corners_sorted = corners[indices][::-1]

    left_uppermost_index = np.argmin(np.sum(corners_sorted, axis=1))
    shifted_corner_points = np.roll(
        corners_sorted, -left_uppermost_index, axis=0
    )

    return shifted_corner_points


def get_point_angles_in_deg(points):
    # The vectors are differences of coordinates
    # a points into the point, b out of the point
    a = points - numpy.roll(points, 1, axis=0)
    b = numpy.roll(a, -1, axis=0)  # same but shifted

    # Calculate length of those vectors
    aLengths = numpy.linalg.norm(a, axis=1)
    bLengths = numpy.linalg.norm(b, axis=1)

    # Calculate length of the cross product
    # Since 2D (not 3D) cross product
    # can't result in a vector, just its z-component
    crossproducts = numpy.cross(a, b) / aLengths / bLengths

    angles = numpy.arcsin(crossproducts)

    return angles / numpy.pi * 180


def get_shape_of_fixed_image(corners: numpy.ndarray) -> Tuple[int, int, int]:
    # TODO: Use correct algorithm as described in the readme

    def maximum(a, b):
        return a if a > b else b

    top_edge_length = numpy.linalg.norm(corners[0] - corners[1])
    bottom_edge_length = numpy.linalg.norm(corners[2] - corners[3])
    width = int(maximum(top_edge_length, bottom_edge_length))

    left_edge_length = numpy.linalg.norm(corners[0] - corners[3])
    right_edge_length = numpy.linalg.norm(corners[1] - corners[2])
    height = int(maximum(left_edge_length, right_edge_length))

    return (height, width, 1)


def get_fixed_image(image, detected_corners):
    shape_of_fixed_image = get_shape_of_fixed_image(detected_corners)
    corners_of_fixed_image = get_img_corners(shape_of_fixed_image)
    projectiveTransform = transform.ProjectiveTransform()
    # Flip coordinates as estimate expects (x, y), but images are (row, column)
    projectiveTransform.estimate(
        numpy.fliplr(numpy.array(corners_of_fixed_image)),
        numpy.fliplr(numpy.array(detected_corners)),
    )

    return transform.warp(
        image,
        projectiveTransform,
        output_shape=shape_of_fixed_image,
        mode="reflect",
    )


def binarize(image, debugger, method="sauvola"):
    radius = 3

    gray_image = rgb2gray(image)
    debugger.save("gray_image", gray_image)
    binarized_image = None

    if method == "sauvola":
        window_size = 3  # Minimal window size
        window_size += image.size // 2**20  # Set relative to image size
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

    # elif method == 'adaptive':
    #     binarized_image = gray_image > threshold_adaptive(image, radius)

    elif method == "niblack":
        sigma = image.size // 2**17

        thresh_niblack = skimage.filters.threshold_niblack(
            image,
            window_size=radius,
            k=0.08,
        )
        binarized_image = image > thresh_niblack

    elif method == "gauss-diff":
        sigma = gray_image.size // (2**16)
        high_frequencies = numpy.subtract(
            gray_image,
            gaussian(
                image=gray_image,
                sigma=sigma,
            ),
        )
        thresh = threshold_otsu(high_frequencies)
        binarized_image = high_frequencies > thresh

    elif method == "local-otsu":
        print("TODO")
        # warped_image_ubyte = img_as_ubyte(image)
        # selem = disk(radius)
        # local_otsu = rank.otsu(warped_image_ubyte, selem)
        # threshold_global_otsu = threshold_otsu(warped_image_ubyte)
        # binary_otsu = warped_image_ubyte >= local_otsu

    else:
        raise TypeError(f"{method} is no supported binarization method")

    debugger.save("binarized_image", binarized_image)

    return binarized_image


def clear(binary_image, debugger):
    """
    Remove noise from border
    """
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
        util.img_as_ubyte(image), morphology.disk(25)
    )
    debugger.save(f"eroded_{image_name}", eroded_image)
    return eroded_image


def get_doc_corners(debugger, output_base_path, image, **kwargs):
    # debug = kwargs.get("debug", False)
    image_marked_path = kwargs.get("image_marked_path")
    intermediate_height = 256

    if image_marked_path:
        image_marked = imageio.imread(image_marked_path, rotate=True)

        # TODO: Scale image *before* doing any computations

        image_gray = rgb2gray(image)
        image_marked_gray = rgb2gray(image_marked)

        # Use value > 0 in range 0 <= x <= 1 to ignore JPEG artifacts
        diff_corner_image = abs(image_gray - image_marked_gray) > 0.05
        debugger.save("diff_corner", diff_corner_image)

        blobs = feature.blob_doh(
            image=diff_corner_image,
            min_sigma=5,
        )

        detected_corners = numpy.delete(blobs, 2, 1)
        corners_normalized = get_sorted_corners(
            image.shape,
            detected_corners,
        )

        if not corners_normalized:
            logging.warn("No corners detected")
            return image

    else:
        scale_ratio = intermediate_height / image.shape[0]

        resized_image = transform.resize(
            image,
            output_shape=(
                intermediate_height,
                # TODO: Scale all images to square size
                int(image.shape[1] * scale_ratio),
            ),
            mode="reflect",
            anti_aliasing=True,
        )
        debugger.save("resized", img_as_ubyte(resized_image))

        resized_gray_image = rgb2gray(resized_image)
        debugger.save("resized_gray", img_as_ubyte(resized_gray_image))

        blurred = gaussian(resized_gray_image, sigma=1)
        debugger.save("blurred", img_as_ubyte(blurred))

        markers = numpy.zeros_like(resized_gray_image, dtype=int)
        markers[0, :] = 1  # Top row
        markers[-1, :] = 1  # Bottom row
        markers[:, 0] = 1  # Left column
        markers[:, -1] = 1  # Right column
        center = (
            resized_gray_image.shape[0] // 2,
            resized_gray_image.shape[1] // 2,
        )
        markers[center] = 2

        elevation_map = sobel(blurred)

        # Flatten elevation map at seed
        # to avoid being trapped in a local minimum
        rows, columns = draw.disk(center, 16)
        elevation_map[rows, columns] = 0.0
        debugger.save(
            "elevation_map",
            exposure.rescale_intensity(img_as_ubyte(elevation_map)),
        )

        segmented_image = watershed(image=elevation_map, markers=markers)

        region_count = len(numpy.unique(segmented_image))

        if region_count != 2:
            logging.error(f"Expected 2 regions and not {region_count}")
            return image

        debugger.save(
            "segmented",
            img_as_ubyte(
                label2rgb(segmented_image, image=resized_gray_image),
            ),
        )

        segmented_relabeled = segmented_image
        segmented_relabeled[segmented_image == 1] = 0
        segmented_relabeled[segmented_image == 2] = 1

        # `img_as_bool` does not work here
        segmented_closed = segmented_relabeled.astype(bool)

        closing_diameter = 25
        pad_width = 2 * closing_diameter

        # Add border to avoid connection with image boundaries
        segmented_closed_border = numpy.pad(
            segmented_closed,
            pad_width=pad_width,
            mode="constant",
            constant_values=False,
        )

        segmented_closed_border = morphology.binary_closing(
            segmented_closed_border,
            morphology.disk(closing_diameter),
        )
        # Remove border
        segmented_closed = segmented_closed_border[
            pad_width:-pad_width,
            pad_width:-pad_width,
        ]

        # Convert False/True to 0/1
        debugger.save("segmented_closed", img_as_ubyte(segmented_closed))

        # Use Foerstner corner detector
        # as with Harris detector corners are shifted inwards
        w, q = corner_foerstner(segmented_closed, sigma=2)
        accuracy_thresh = 0.5
        roundness_thresh = 0.3
        foerstner = (q > roundness_thresh) * (w > accuracy_thresh) * w
        foerstner_corners = corner_peaks(foerstner, min_distance=1)
        logging.info(f"foerstner_corners: {foerstner_corners}")

        # Render corners
        empty_img = numpy.zeros_like(segmented_closed)
        empty_img[foerstner_corners[:, 0], foerstner_corners[:, 1]] = 1
        debugger.save("corner_foerstner", img_as_ubyte(empty_img))

        foerstner_corners_sorted = get_sorted_corners(
            segmented_closed.shape,
            foerstner_corners,
        )

        logging.info(f"foerstner corners sorted: {foerstner_corners_sorted}")

        point_angles_in_deg = get_point_angles_in_deg(
            foerstner_corners_sorted,
        )
        logging.info(f"point_angles_in_deg: {point_angles_in_deg}")

        point_angles_abs = numpy.abs(point_angles_in_deg)

        # Get the indices of the sorted angles in descending order
        point_angles_abs_sorted = numpy.argsort(point_angles_abs)[::-1]
        logging.info(f"point_angles_abs_sorted {point_angles_abs_sorted}")

        top_4_indices = point_angles_abs_sorted[:4]
        # Sort the top 4 indices to maintain the original order
        # in foerstner_corners_sorted
        sorted_top_4_indices = np.sort(top_4_indices)

        # Select the top 4 corners with the largest angle
        # while maintaining their original order
        corners_final = foerstner_corners_sorted[sorted_top_4_indices]

        logging.info(f"corners_final: {corners_final}")

        rows, columns = draw.polygon_perimeter(
            corners_final[:, 0],
            corners_final[:, 1],
        )
        image_simplified = numpy.copy(segmented_closed).astype(int)
        image_simplified[rows, columns] = 4
        debugger.save(
            "simplified",
            img_as_ubyte(
                label2rgb(
                    image_simplified,
                    image=resized_image,  # Overlay over original image
                    bg_label=0,
                )
            ),
        )

        if not numpy.any(corners_final):
            return image

        corners_normalized = numpy.divide(corners_final, scale_ratio)

        # TODO: Compare with values stored in json files

        return corners_normalized


def setup_logger(output_base_path):
    os.makedirs(output_base_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_base_path, "0-log.txt"),
        level=logging.DEBUG,
        format=" - ".join(
            [
                "%(asctime)s",
                "%(pathname)s:%(lineno)s",
                "%(levelname)s",
                "%(name)s",
                "%(message)s",
            ]
        ),
    )


def transform_image(**kwargs):
    input_image_path = kwargs.get("input_image_path")

    if not input_image_path:
        raise FileNotFoundError(
            f"An input image and not {input_image_path} must be specified"
        )

    output_in_gray = kwargs.get("output_in_gray", False)
    binarization_method = kwargs.get("binarization_method")
    shall_clear_border = not kwargs.get("shall_not_clear_border", False)

    file_name_segments = os.path.splitext(os.path.basename(input_image_path))
    basename = file_name_segments[0]
    output_base_path = os.path.join(
        os.path.dirname(input_image_path),
        basename,
    )

    debug = kwargs.get("debug", False)

    # TODO: Accept lambda function which is only executed during debugging
    debugger = ImageDebugger(
        level="debug" if debug else "",
        base_path=output_base_path,
    )

    if debug:
        setup_logger(output_base_path)

    image = imageio.imread(input_image_path, rotate=True)

    corners = get_doc_corners(debugger, output_base_path, image)

    dewarped_image = get_fixed_image(image, corners)
    debugger.save("dewarped", img_as_ubyte(dewarped_image))

    if output_in_gray:
        grayscale_image = rgb2gray(dewarped_image)
        image_norm_intensity = exposure.rescale_intensity(grayscale_image)
        debugger.save("normalized_intensity", image_norm_intensity)
        transformed_image = image_norm_intensity

    elif binarization_method:
        binarized_image = binarize(
            image=dewarped_image,
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

        transformed_image = denoised_image

    # TODO: elif is_book:

    else:
        transformed_image = dewarped_image

    random_string = (
        base64.b64encode(os.urandom(3))
        .decode("utf-8")
        .replace("+", "-")
        .replace("/", "_")
    )
    output_image_path = (
        kwargs.get("output_image_path")
        or f"{output_base_path}-fixed_{random_string}.png"
    )

    if not debug:
        imageio.imwrite(
            output_image_path,
            img_as_ubyte(transformed_image),
        )


def print_corners(**kwargs):
    input_image_path = kwargs.get("input_image_path")

    if not input_image_path:
        raise FileNotFoundError(
            f"An input image and not {input_image_path} must be specified"
        )

    file_name_segments = os.path.splitext(os.path.basename(input_image_path))
    basename = file_name_segments[0]
    output_base_path = os.path.join(
        os.path.dirname(input_image_path),
        basename,
    )

    debug = kwargs.get("debug", False)

    # TODO: Accept lambda function which is only executed during debugging
    debugger = ImageDebugger(
        level="debug" if debug else "",
        base_path=output_base_path,
    )

    if debug:
        setup_logger(output_base_path)

    image = imageio.imread(input_image_path, rotate=True)

    doc_corners = get_doc_corners(debugger, output_base_path, image)

    # Origin is top left corner
    corner_dicts = [{"x": corner[1], "y": corner[0]} for corner in doc_corners]
    json_str = json.dumps(corner_dicts)
    print(json_str)
