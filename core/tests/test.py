import matplotlib
import matplotlib.pyplot as plt

import numpy as np
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


image_shape = (256, 256)
spot_radius = 8
border_width = 8


def scale_image(new_shape, image):
    return transform.resize(
        image,
        output_shape=tuple(value - border_width for value in new_shape),
        mode='reflect',
    )


def get_basin_mask(image):
    """
    Get basin mask for watershed algorithm
    """
    shape = np.shape(image)
    basinMask = np.zeros(shape)
    basinMask[(0, 0)] = 1
    basinMask[tuple(value // 2 for value in shape)] = 2
    return basinMask


def level_image(border_width, spot_radius, elevation_image):
    """
    Level the elevation map at the center and around the border
    to avoid being trapped in a local minimum during flooding
    """
    shape = np.shape(elevation_image)
    shape_padded = tuple(value + (2 * border_width) for value in shape)
    elevation_padded = util.pad(elevation_image, border_width, 'constant')
    center = tuple(value / 2 for value in shape_padded)
    rows, columns = draw.circle(center[0], center[1], spot_radius)
    elevation_padded[rows, columns] = 0.0
    return elevation_padded


images = io.ImageCollection(
    load_pattern='../../examples/photos/*/in.jpg',
    conserve_memory=True,
)

offset = 1
rand_img_index = np.random.randint(0, len(images) - offset)
images = images[rand_img_index:rand_img_index + offset]
images_gray = map(color.rgb2gray, images)
images_scaled = map(
    lambda img: scale_image(image_shape, img),
    images_gray,
)
images_blured = map(filters.gaussian, images_scaled)
images_elevation = map(filters.sobel, images_blured)
images_leveled = map(
    lambda img: level_image(border_width, spot_radius, img),
    images_elevation,
)
images_segmented = map(
    lambda img: segmentation.watershed(img, get_basin_mask(img)),
    images_leveled,
)

images_final = list(images_segmented)

# io.imshow_collection(images_scaled, plugin='matplotlib')
# io.show()
io.imshow_collection(images_final, plugin='matplotlib')
io.show()
