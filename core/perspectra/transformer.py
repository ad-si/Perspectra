import os
import base64
from pathlib import Path
import logging

import imageio
import numpy
import skimage
from skimage import (
    color,
    draw,
    exposure,
    feature,
    filters,
    io,
    measure,
    morphology,
    segmentation,
    transform,
    util,
)
from skimage.color import rgb2gray, label2rgb
from skimage.exposure import rescale_intensity
from skimage.draw import circle, circle_perimeter
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import (
    rank, sobel, gaussian, median,
    threshold_adaptive, threshold_otsu, threshold_sauvola
)
from skimage.morphology import watershed, disk
from skimage.util import img_as_ubyte

from .multipass_cleaner import remove_noise


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
        if self.level != 'debug':
            return
        self.step_counter += 1
        image_path = os.path.join(
            self.base_path,
            f'{self.step_counter}-{name}.png',
        )
        imageio.imwrite(image_path, image)
        logging.info(f'Stored image: {image_path}')
        return self


def load_image(file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    return io.imread(file_path)


def vertices_to_edges(vertices):
    edges = []
    for index, coordinate in enumerate(vertices):
        if index != (len(vertices) - 1):
            edges.append([coordinate, vertices[index + 1]])
    return edges


def simplify_polygon(vertices, targetCount=4):
    """
    Merge globally shortest edge with shortest neighbor
    """
    if not numpy.array_equal(vertices[0], vertices[-1]):
        raise ValueError(f'First vertex ({vertices[0]}) \
            and last ({vertices[-1]}) must be the same')

    edges = numpy.array(vertices_to_edges(vertices))

    while len(edges) > targetCount:
        edges_lengths = [
            numpy.linalg.norm(edge[0] - edge[1])
            for edge in edges
        ]
        edge_min_index = numpy.argmin(edges_lengths)
        edge_prev_length = edges_lengths[edge_min_index - 1]
        edge_next_length = numpy.take(
            edges_lengths,
            edge_min_index + 1,
            mode='wrap',
        )

        if edge_prev_length < edge_next_length:
            # Merge with previous edge
            edges[edge_min_index][0] = edges[edge_min_index - 1][0]
            edges = numpy.delete(edges, edge_min_index - 1, axis=0)
            edges_lengths = numpy.delete(edges_lengths, edge_min_index - 1)
        else:
            # Merge with next edge
            edges[edge_min_index][1] = \
                edges[(edge_min_index + 1) % len(edges)][1]
            edges = numpy.delete(
                edges,
                (edge_min_index + 1) % len(edges),
                axis=0
            )
            edges_lengths = numpy.delete(
                edges_lengths,
                (edge_min_index + 1) % len(edges)
            )

    # Re-add first vertex to close polygon
    vertices_new = numpy.append(edges[:, 0], [edges[0][0]], axis=0)
    return vertices_new


def get_corners(shape):
    rows = shape[0]
    colums = shape[1]
    return [
        (0, 0),
        (0, colums - 1),
        (rows - 1, colums - 1),
        (rows - 1, 0),
    ]


def get_sorted_corners(corners):
    # Sort-order: top-left, top-right, bottom-right, bottom-left

    if not numpy.any(corners):
        return None

    # Empty placeholder array
    sorted_corners = numpy.zeros((4, 2))

    sum_row_column = corners.sum(axis=1)
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


def get_shape_of_fixed_image(corners):
    # TODO: Use correct algorithm as described in the readme

    top_edge_length = numpy.linalg.norm(corners[0] - corners[1])
    bottom_edge_length = numpy.linalg.norm(corners[2] - corners[3])
    width = int(max(top_edge_length, bottom_edge_length))

    left_edge_length = numpy.linalg.norm(corners[0] - corners[3])
    right_edge_length = numpy.linalg.norm(corners[1] - corners[2])
    height = int(max(left_edge_length, right_edge_length))

    return (height, width, 1)


def get_fixed_image(image, detected_corners):
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
        output_shape=shape_of_fixed_image,
        mode='reflect',
    )


def binarize(image, debugger, method='sauvola'):
    radius = 3

    gray_image = rgb2gray(image)
    debugger.save('gray_image', gray_image)

    if method == 'sauvola':
        window_size = 3  # Minimal window size
        window_size += image.size // 2 ** 20  # Set relative to image size
        window_size += 1 if (window_size % 2 == 0) else 0  # Must always be odd
        logging.info(f'window_size: {window_size}')

        thresh_sauvola = numpy.nan_to_num(threshold_sauvola(
            image=gray_image,
            window_size=window_size,
            k=0.3,  # Attained through experimentation
        ))
        debugger.save('thresh_sauvola', thresh_sauvola)
        binarized_image = gray_image > thresh_sauvola

    elif method == 'adaptive':
        binarized_image = gray_image > threshold_adaptive(image, radius)

    elif method == 'niblack':
        sigma = image.size // 2 ** 17

        thresh_niblack = skimage.filters.threshold_niblack(
            image,
            window_size=radius,
            k=0.08,
        )
        binarized_image = image > thresh_niblack

    elif method == 'gauss-diff':
        sigma = gray_image.size // (2 ** 16)
        high_frequencies = np.subtract(
            gray_image,
            gaussian(image=gray_image, sigma=sigma,)
        )
        thresh = threshold_otsu(high_frequencies)
        binarized_image = high_frequencies > thresh

    elif method == 'local-otsu':
        warped_image_ubyte = img_as_ubyte(image)
        selem = disk(radius)
        local_otsu = rank.otsu(warped_image_ubyte, selem)
        threshold_global_otsu = threshold_otsu(warped_image_ubyte)
        binary_otsu = warped_image_ubyte >= local_otsu

    else:
        raise TypeError(f'{method} is no supported binarization method')

    debugger.save('binarized_image', binarized_image)

    return binarized_image


def clear(binary_image, debugger):
    """
    Remove noise from border
    """
    inverted_image = util.invert(binary_image)
    inverted_cleared_image = segmentation.clear_border(inverted_image)
    cleared_image = util.invert(inverted_cleared_image)
    debugger.save('cleared_image', cleared_image)
    return cleared_image


def denoise(binary_image, debugger):
    inverted_image = util.invert(binary_image)
    inverted_denoised_image = remove_noise(inverted_image)
    denoised_image = util.invert(inverted_denoised_image)
    debugger.save('denoised_image', denoised_image)

    return denoised_image


def erode(image, image_name, debugger):
    eroded_image = morphology.erosion(
        util.img_as_ubyte(image),
        morphology.disk(25)
    )
    debugger.save(f'eroded_{image_name}', eroded_image)
    return eroded_image


def transform_image(**kwargs):
    input_image_path = kwargs.get('input_image_path')

    if not input_image_path:
        raise FileNotFoundError(
            f'An input image and not {input_image_path} must be specified'
        )

    output_in_gray = kwargs.get('output_in_gray', False)
    binarization_method = kwargs.get('binarization_method')
    shall_clear_border = not kwargs.get('shall_not_clear_border', False)
    debug = kwargs.get('debug', False)
    image_marked_path = kwargs.get('image_marked_path')
    adaptive = kwargs.get('adaptive')
    intermediate_height = 500

    file_name_segments = os.path.splitext(os.path.basename(input_image_path))
    basename = file_name_segments[0]
    extension = file_name_segments[1]
    random_string = (base64
                     .b64encode(os.urandom(3))
                     .decode('utf-8')
                     .replace('+', '-')
                     .replace('/', '_')
                     )

    output_base_path = os.path.join(
        os.path.dirname(input_image_path),
        basename
    )

    # Settings for debugging
    if debug:
        os.makedirs(output_base_path, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(output_base_path, '0-log.txt'),
            level=logging.DEBUG,
            format=' - '.join([
                '%(asctime)s',
                '%(pathname)s:%(lineno)s',
                '%(levelname)s',
                '%(name)s',
                '%(message)s',
            ]),
        )

    debugger = ImageDebugger(
        level='debug' if debug else '',
        base_path=output_base_path,
    )

    output_image_path = (
        kwargs.get('output_image_path') or
        f'{output_base_path}-fixed_{random_string}.png'
    )

    def get_transformed_image():
        if input_image_path.lower().endswith(('jpg', 'jpeg')):
            image = imageio.imread(input_image_path, exifrotate=True)
        else:
            # It's unclear under which circumstances gamma gets corrected
            # => always ignore it
            image = imageio.imread(input_image_path, ignoregamma=True)

        if image_marked_path:
            if image_marked_path.endswith(('jpg', 'jpeg')):
                image_marked = imageio.imread(
                    image_marked_path,
                    exifrotate=True
                )
            else:
                # Can't replicate when gamma gets corrected => always ignore it
                image_marked = imageio.imread(
                    image_marked_path,
                    ignoregamma=True
                )

            # TODO: Scale image before doing any computations

            image_gray = rgb2gray(image)
            image_marked_gray = rgb2gray(image_marked)

            # Use value > 0 in range 0 <= x <= 1 to ignore JPEG artifacts
            diff_corner_image = abs(image_gray - image_marked_gray) > 0.05
            debugger.save('diff_corner', diff_corner_image)

            blobs = feature.blob_doh(
                image=diff_corner_image,
                min_sigma=5,
            )

            detected_corners = numpy.delete(blobs, 2, 1)
            corners_normalized = get_sorted_corners(detected_corners)

            if not numpy.any(corners_normalized):
                logging.warn('No corners detected')
                return image

        else:
            scale_ratio = intermediate_height / image.shape[0]

            resized_image = transform.resize(
                image,
                output_shape=(
                    intermediate_height,
                    int(image.shape[1] * scale_ratio)
                ),
                mode='reflect',
            )
            debugger.save('resized', resized_image)

            image_corners = get_corners(resized_image.shape)

            resized_gray_image = rgb2gray(resized_image)
            debugger.save('resized_gray', resized_gray_image)

            blurred = gaussian(resized_gray_image, sigma=1)
            debugger.save('blurred', blurred)

            markers = numpy.zeros_like(resized_gray_image)
            center = (
                resized_gray_image.shape[0] // 2,
                resized_gray_image.shape[1] // 2,
            )
            markers[(0, 0)] = 1
            markers[center] = 2

            elevation_map = sobel(blurred)

            # Flatten elevation map at seed
            # to avoid being trapped in a local minimum
            rows, columns = circle(center[0], center[1], 10)
            elevation_map[rows, columns] = 0.0
            debugger.save('elevation_map', elevation_map)

            segmented_image = watershed(image=elevation_map, markers=markers)

            region_count = len(numpy.unique(segmented_image))

            if region_count != 2:
                logging.error(f'Expected 2 regions and not {region_count}')
                return image

            debugger.save('segmented', segmented_image)

            segmented_relabeled = segmented_image
            segmented_relabeled[segmented_image == 1] = 0
            segmented_relabeled[segmented_image == 2] = 1
            segmented_closed = segmented_relabeled.astype(bool)

            closing_diameter = 25
            pad_width = 2 * closing_diameter

            # Add border to avoid connection with image boundaries
            segmented_closed_border = numpy.pad(
                segmented_closed,
                pad_width=pad_width,
                mode='constant',
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
            debugger.save('segmented_closed', segmented_closed.astype(int))

            def get_harris_peaks(image, sigma=5, k=0.05):
                img_harris = corner_harris(image, sigma=sigma, k=k)

                # `min_distance` prevents `image_corners` from being included
                peaks = corner_peaks(img_harris, min_distance=4)
                draw.set_color(img_harris, numpy.transpose(peaks), 0)
                debugger.save('harris_corner', img_harris)

                return peaks

            corners_second_pass = get_harris_peaks(
                segmented_closed, sigma=4, k=0.04)

            corners_orientation = feature.corner_orientations(
                segmented_closed,
                corners_second_pass,
                morphology.octagon(3, 2),
            )
            logging.info(f'corners_orientation: {corners_orientation}')

            indices = numpy.argsort(corners_orientation)
            corners_sorted = corners_second_pass[indices]
            logging.info(f'corners_sorted: {corners_sorted}')

            polygon_simple = simplify_polygon(
                numpy.append(corners_sorted, [corners_sorted[0]], axis=0),
                targetCount=4,
            )
            logging.info(f'Simplified coordinates: {polygon_simple}')

            rows, columns = draw.polygon_perimeter(
                polygon_simple[:, 0],
                polygon_simple[:, 1],
            )
            image_simplified = numpy.copy(segmented_image)
            image_simplified[rows, columns] = 4
            debugger.save('simplified', label2rgb(
                image_simplified,
                image=resized_image,  # Overlay over original image
                bg_label=0,
            ))

            sorted_corners = get_sorted_corners(polygon_simple)
            logging.info(f'Sorted corners: {sorted_corners}')

            if not numpy.any(sorted_corners):
                return image

            corners_normalized = numpy.divide(sorted_corners, scale_ratio)

        dewarped_image = get_fixed_image(image, corners_normalized)
        debugger.save('dewarped', dewarped_image)

        # TODO: if is_book:

        if output_in_gray:
            grayscale_image = rgb2gray(dewarped_image)
            image_norm_intensity = exposure.rescale_intensity(grayscale_image)
            debugger.save('normalized_intensity', image_norm_intensity)
            return image_norm_intensity

        if binarization_method:
            binarized_image = binarize(
                image=dewarped_image,
                method=binarization_method,
                debugger=debugger,
            )
            if shall_clear_border:
                cleared_image = clear(binarized_image, debugger)
                erode(cleared_image, 'cleared', debugger)
                denoised_image = denoise(cleared_image, debugger)
            else:
                erode(binarized_image, 'binarized', debugger)
                denoised_image = denoise(binarized_image, debugger)

            erode(denoised_image, 'denoised', debugger)

            return denoised_image

        return dewarped_image

    transformed_image = get_transformed_image()

    if not debug:
        imageio.imwrite(output_image_path, transformed_image)
