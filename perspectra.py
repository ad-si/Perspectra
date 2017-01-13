import os

import numpy
from numpy import linalg

import skimage
from skimage import filters, io, transform
from skimage.color import rgb2gray
from skimage.draw import circle, circle_perimeter
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import rank, sobel, gaussian, threshold_adaptive, threshold_otsu
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


image = load_image('images/document.png')
gray_image = rgb2gray(image)

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

warped_image = get_fixed_image(gray_image, scaled_corners)

# Must always be odd (+ 1)
radius = (image.size // 2 ** 17) + 1
sigma = image.size // 2 ** 17
print(radius, sigma)
binary_adaptive = threshold_adaptive(warped_image, radius)

thresh_niblack = skimage.filters.threshold_niblack(
    warped_image,
    window_size = radius,
    k=0.08,
)
binary_niblack = warped_image > thresh_niblack

thresh_sauvola = skimage.filters.threshold_sauvola(
    warped_image,
    window_size = radius,
    k=0.04
)
binary_sauvola = warped_image > thresh_sauvola

selem = disk(radius)
warped_image_ubyte = img_as_ubyte(warped_image)
local_otsu = rank.otsu(warped_image_ubyte, selem)
threshold_global_otsu = threshold_otsu(warped_image_ubyte)
binary_otsu = warped_image_ubyte >= local_otsu

high_frequencies = warped_image - gaussian(warped_image, sigma=sigma)
thresh_adi = threshold_otsu(high_frequencies)
binary_adi = high_frequencies > thresh_adi

binary_wtf = warped_image - skimage.filters.rank.median(warped_image, disk(radius))

fig, ((pos0, pos1), (pos2, pos3), (pos4, pos5),
  (pos6, pos7), (pos8, pos9)) = pyplot.subplots(
    nrows=5,
    ncols=2,
    figsize=(10, 12)
)

pos0.imshow(resized_image)
pos0.set_title('1. Resized image')

pos1.imshow(scaled_gray_image)
pos1.set_title('2. Luminance')

pos2.imshow(blurred)
pos2.set_title('3. Blurred with Gaussian filter')

pos3.imshow(elevation_map)
pos3.set_title('4. Edge detection with Sobel filter')

pos4.imshow(segmented_image)
pos4.set_title('5. Segmentation with watershed algorithm')

pos5.imshow(harris_image)
pos5.plot(sorted_corners[:, 1], sorted_corners[:, 0], '+r', markersize=10)
pos5.set_title('6. Harris corner image with corner peaks')

pos6.imshow(resized_image)
pos6.plot(sorted_corners[:, 1], sorted_corners[:, 0], '+r', markersize=10)
pos6.set_title('7. Original image with detected corners')

pos7.imshow(warped_image)
pos7.set_title('8. Corrected perspective and corrected size')

pos8.imshow(binary_otsu, cmap=pyplot.cm.gray)
pos8.set_title('9. Binarize with adaptive version of otsu\'s method')

pos9.imshow(binary_sauvola, cmap=pyplot.cm.gray)
pos9.set_title('10. Binarize with adaptive version of otsu\'s method')

# cursor = Cursor(ax0, color='red', linewidth=1)


# figure = pyplot.figure()
# pyplot.imshow(image)
# figure.canvas.mpl_connect('button_press_event', onclick)

pyplot.show()
