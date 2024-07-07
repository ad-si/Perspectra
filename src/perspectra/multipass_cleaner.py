from typing import (List, Any)
import numpy
import logging
from skimage import morphology


def remove_noise(
        original_img: List[List[int]],
        passes: int=7,
        images: List[Any]=[]
        ) -> List[List[int]]:
    """
    Larger blobs must be increasingly separated to be labeled as noise
    """
    cleaned_orig = original_img
    cleaned_eroded = original_img
    radius_step = 2

    for index in range(passes):
        cummulative_dilation_size = radius_step ** index
        cummulative_disk = morphology.disk(cummulative_dilation_size)
        logging.info(f'Cummulative dilation size: {cummulative_dilation_size}')

        current_dilation_size = radius_step ** (index - 1) \
            if index > 0 \
            else 1
        # dilation_disk = morphology.disk(current_dilation_size)
        logging.info(f'Current dilation size: {current_dilation_size}')

        # Noise blobs with up to 80 % more area
        # than the structuring element will get deleted
        max_noise_size = numpy.count_nonzero(cummulative_disk) * 1.5
        logging.info(f'Maximum noise size: {max_noise_size}')

        eroded = morphology.dilation(
            cleaned_eroded,
            footprint=morphology.disk(current_dilation_size)
        )

        if images:
            images.append((f'eroded {index}', eroded))

        cleaned_eroded = morphology.remove_small_objects(
            eroded,
            max_noise_size
        )

        if images:
            images.append((f'cleaned eroded {index}', cleaned_eroded))

        cleaned_orig = numpy.logical_and(cleaned_orig, cleaned_eroded)

        logging.info(f'Finished cleaning pass {index}\n')

    return cleaned_orig
