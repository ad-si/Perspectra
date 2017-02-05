import os
import base64

import numpy
from numpy import linalg

import skimage
from skimage import filters, io, transform
from skimage.color import rgb2gray
from skimage.draw import circle, circle_perimeter
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import (
    rank, sobel, gaussian,
    threshold_adaptive, threshold_otsu
)
from skimage.morphology import watershed, disk
from skimage.util import img_as_ubyte

from matplotlib import pyplot
from matplotlib.widgets import Cursor


def load_image (file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    return io.imread(file_path)


def get_corners (shape):
    rows = shape[0]
    colums = shape[1]
    return [
        (0, 0),
        (0, colums - 1),
        (rows - 1, colums - 1),
        (rows - 1, 0),
    ]


def get_sorted_corners (corners):
    # Sort-order: top-left, top-right, bottom-right, bottom-left

    # Empty placeholder array
    sorted_corners = numpy.zeros((4, 2))

    sum_row_column = corners.sum(axis = 1)
    # Top-left => smallest sum
    sorted_corners[0] = corners[numpy.argmin(sum_row_column)]
    # Bottom-right => largest sum
    sorted_corners[2] = corners[numpy.argmax(sum_row_column)]

    column_minus_row = numpy.diff(corners)
    # Top-right => largest difference
    sorted_corners[1] = corners[numpy.argmax(column_minus_row)]
    # Bottom-right => smallest difference
    sorted_corners[3] = corners[numpy.argmin(column_minus_row)]

    return sorted_corners


def get_shape_of_fixed_image (corners):
    # TODO: Use correct algorithm as described in the readme

    top_edge_length = linalg.norm(corners[0] - corners[1])
    bottom_edge_length = linalg.norm(corners[2] - corners[3])
    # Reduce aliasing by scaling the output image (the * 2)
    width = int(2 * max(top_edge_length, bottom_edge_length))

    left_edge_length = linalg.norm(corners[0] - corners[3])
    right_edge_length = linalg.norm(corners[1] - corners[2])
    height = int(2 * max(left_edge_length, right_edge_length))

    return (height, width, 1)


def get_fixed_image (image, detected_corners):
    image_corners = get_corners(image.shape)
    shape_of_fixed_image = get_shape_of_fixed_image(detected_corners)
    corners_of_fixed_image = get_corners(shape_of_fixed_image)
    projectiveTransform = transform.ProjectiveTransform()
    # Flip coordinates as estimate expects (x, y), but images are (row, column)
    projectiveTransform.estimate(
        numpy.fliplr(numpy.array(corners_of_fixed_image)),
        numpy.fliplr(numpy.array(detected_corners))
    )

    return transform.warp(
        image,
        projectiveTransform,
        output_shape=shape_of_fixed_image
    )


def onclick (event):
    print(
        'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        (event.button, event.x, event.y, event.xdata, event.ydata)
    )


def render_processing_steps (**kwargs):
    sorted_corners = kwargs.get('sorted_corners')

    fig, ((pos0, pos1), (pos2, pos3), (pos4, pos5),
      (pos6, pos7), (pos8, pos9)) = pyplot.subplots(
        nrows=5,
        ncols=2,
        figsize=(10, 14)
    )

    pos0.set_title('1. Resized image')
    pos0.imshow(kwargs['resized_image'])

    pos1.set_title('2. Luminance')
    pos1.imshow(kwargs['scaled_gray_image'], cmap=pyplot.cm.gray)

    pos2.set_title('3. Blurred with Gaussian filter')
    pos2.imshow(kwargs['blurred'], cmap=pyplot.cm.gray)

    pos3.set_title('4. Edge detection with Sobel filter')
    pos3.imshow(kwargs['elevation_map'])

    pos4.set_title('5. Segmentation with watershed algorithm')
    pos4.imshow(kwargs['segmented_image'], cmap=pyplot.cm.gray)

    pos5.set_title('6. Harris corner image with corner peaks')
    pos5.imshow(kwargs['harris_image'])
    pos5.plot(sorted_corners[:, 1], sorted_corners[:, 0], '+r', markersize=10)

    pos6.set_title('7. Original image with detected corners')
    pos6.imshow(kwargs['resized_image'])
    pos6.plot(sorted_corners[:, 1], sorted_corners[:, 0], '+r', markersize=10)

    pos7.set_title('8. Corrected perspective and corrected size')
    pos7.imshow(kwargs['warped_image'], cmap=pyplot.cm.gray)

    pos8.set_title('9. Binarize with adaptive version of Otsu\'s method')
    pos8.imshow(kwargs['binary_otsu'], cmap=pyplot.cm.gray)

    pos9.set_title('10. Binarize with adaptive version of Otsu\'s method')
    pos9.imshow(kwargs['binary_otsu'], cmap=pyplot.cm.gray)


def transform_image (**kwargs):
    abs_input_image_path = kwargs.get('input_image_path')

    if not abs_input_image_path:
        raise FileNotFoundError(
            f'An input image and not {abs_input_image_path} must be specified'
        )

    basename = os.path.splitext(os.path.basename(abs_input_image_path))[0]
    random_string = (base64
        .b64encode(os.urandom(3))
        .decode('utf-8')
        .replace('+', '-')
        .replace('/', '_')
    )

    output_image_path = kwargs.get('output_image_path') or \
        os.path.join(
            os.path.dirname(abs_input_image_path),
            f'{basename}-fixed_{random_string}.png'
        )

    output_in_gray = kwargs.get('output_in_gray', False)
    output_binary = kwargs.get('output_binary', False)
    binary_algorithms = kwargs.get('binary_algorithms', [])
    debug = kwargs.get('debug', False)
    input_image_path = kwargs.get('input_image_path')
    adaptive = kwargs.get('adaptive')
    image = io.imread(abs_input_image_path)

    intermediate_height = 500
    scale_ratio = intermediate_height / image.shape[0]
    resized_image = transform.resize(
        image,
        output_shape=(intermediate_height, int(image.shape[1] * scale_ratio))
    )
    image_corners = get_corners(resized_image.shape)

    scaled_gray_image = rgb2gray(resized_image)
    blurred = gaussian(scaled_gray_image, sigma=1)
    elevation_map = sobel(blurred)

    markers = numpy.zeros_like(scaled_gray_image)
    center = (scaled_gray_image.shape[0] // 2, scaled_gray_image.shape[1] // 2)
    markers[(0, 0)] = 1
    markers[center] = 2

    segmented_image = watershed(image=elevation_map, markers=markers)
    harris_image = corner_harris(segmented_image, sigma=5)
    # min_distance prevents image_corners from being included
    detected_corners = corner_peaks(harris_image, min_distance=5)
    # detected_corners = [(180, 220), (170, 1300), (740, 1400), (790, 210)]
    sorted_corners = get_sorted_corners(detected_corners)
    scaled_corners = numpy.divide(sorted_corners, scale_ratio)

    warped_image = get_fixed_image(image, scaled_corners)
    fixed_image = warped_image
    gray_warped_image = rgb2gray(warped_image)

    # if output_in_gray or debug:
    #     gray_warped_image = rgb2gray(warped_image)

    if output_binary or debug:
        # Must always be odd (+ 1)
        radius = (image.size // 2 ** 17) + 1
        thresh_sauvola = skimage.filters.threshold_sauvola(
            gray_warped_image,
            window_size = radius,
            k = 0.04
        )
        fixed_image = gray_warped_image > thresh_sauvola

    if adaptive or debug:
        radius = (image.size // 2 ** 17) + 1
        fixed_image = threshold_adaptive(gray_warped_image, radius)

    if ('niblack' in binary_algorithms) or debug:
        radius = (image.size // 2 ** 17) + 1
        sigma = image.size // 2 ** 17

        thresh_niblack = skimage.filters.threshold_niblack(
            gray_warped_image,
            window_size = radius,
            k=0.08,
        )
        fixed_image = gray_warped_image > thresh_niblack

    # if sauvola or debug:
    # selem = disk(radius)
    # warped_image_ubyte = img_as_ubyte(warped_image)
    # local_otsu = rank.otsu(warped_image_ubyte, selem)
    # threshold_global_otsu = threshold_otsu(warped_image_ubyte)
    # binary_otsu = warped_image_ubyte >= local_otsu
    #
    # high_frequencies = warped_image - gaussian(warped_image, sigma=sigma)
    # thresh_adi = threshold_otsu(high_frequencies)
    # binary_adi = high_frequencies > thresh_adi
    #
    # binary_wtf = warped_image - skimage.filters.rank
    #   .median(warped_image, disk(radius))

    if debug:
        render_processing_steps(
            resized_image = resized_image,
            scaled_gray_image = scaled_gray_image,
            blurred = blurred,
            elevation_map = elevation_map,
            segmented_image = segmented_image,
            harris_image = harris_image,
            warped_image = warped_image,
            binary_otsu = fixed_image,
            sorted_corners = sorted_corners,
            # binary_sauvola = binary_sauvola
        )
        pyplot.show()

    else:
        io.imsave(output_image_path, fixed_image)


    # cursor = Cursor(ax0, color='red', linewidth=1)

    # figure = pyplot.figure()
    # pyplot.imshow(image)
    # figure.canvas.mpl_connect('button_press_event', onclick)
