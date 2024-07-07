from typing import (List)
import numpy
from skimage import morphology


def split_pages(image: List[List[int]]) -> List[List[int]]:
    """
    1. Mark pixels in the center as more likely to contain the split
    2. Get vertical Houghlines
    3. Split image
    """

    index = 0
    radius_step = 2
    cummulative_dilation_size = radius_step ** index
    cummulative_disk = morphology.disk(cummulative_dilation_size)
    print(f'Cummulative dilation size: {cummulative_dilation_size}')

    current_dilation_size = radius_step ** (index - 1) \
        if index > 0 \
        else 1
    # dilation_disk = morphology.disk(current_dilation_size)
    print(f'Current dilation size: {current_dilation_size}')

    # Noise blobs with up to 80 % more area
    # than the structuring element will get deleted
    max_noise_size = numpy.count_nonzero(cummulative_disk) * 1.5
    print(f'Maximum noise size: {max_noise_size}')

    # eroded = morphology.dilation(
    #     image,
    #     selem=morphology.disk(current_dilation_size)
    # )

    # if images:
    #     images.append((f'eroded {index}', eroded))

    # cleaned_eroded = morphology.remove_small_objects(
    #     eroded,
    #     max_noise_size
    # )

    # if images:
    #     images.append((f'cleaned eroded {index}', cleaned_eroded))

    # cleaned_orig = numpy.logical_and( image , cleaned_eroded)

    print(f'Finished cleaning pass {index}\n')

    return [] # TODO: [leftPage, rightPage]
