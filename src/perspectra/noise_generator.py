import random
import numpy
from skimage import morphology, util


def add_noise(image):
    # base_radius = 2
    # base_disk = morphology.disk(base_radius)
    total_amount = 0.001
    empty_image = numpy.zeros_like(image)

    noisy_img = util.random_noise(
        image=empty_image,
        mode='salt',
        amount=total_amount,
        rng=123,
    )
    closed_noise = morphology.binary_closing(
        image=noisy_img,
        footprint=morphology.disk(15)
    )
    labeled_noise = morphology.label(closed_noise)
    keep_percentage = 0.05
    number_of_noise_blobs = len(numpy.unique(labeled_noise))
    random.seed(a=123)
    random_blob_labels = random.sample(
        # Start with 1 to not get the background color, which is 0
        range(1, number_of_noise_blobs),
        int(number_of_noise_blobs * keep_percentage),
    )
    filtered_noise = numpy.isin(labeled_noise.flat, random_blob_labels)
    reshaped_noise = filtered_noise.reshape(image.shape)

    return numpy.logical_or(reshaped_noise, image)
