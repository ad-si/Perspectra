from perspectra import binarize
import numpy as np

def test_binarize_grayscale_image():
    debugger = binarize.ImageDebugger(
        level="",
        base_path="",
    )
    white = [255, 255, 255]
    black = [0, 0, 0]
    example_image = [
            [black, black, black, [4, 4, 4], black],
            [black, white, white, white, black ],
            [[8, 8, 8], white, [200, 200, 200], white, black],
            [black, white, white, white, [9, 9, 9]],
            [black, black, [8, 8, 8], black, black],
        ]

    result = binarize.binarize(example_image, debugger)
    assert np.array_equal(result, [
        [False, False, False, False, False],
        [False,  True,  True,  True, False],
        [False,  True,  True,  True, False],
        [False,  True,  True,  True, False],
        [False, False, False, False, False],
    ])
