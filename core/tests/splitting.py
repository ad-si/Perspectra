import math
import numpy
from skimage import io, util, feature, transform, filters, morphology
from matplotlib import pyplot


images = []

original = io.imread('/Users/adrian/Projects/perspectra/examples/handheld-books/2/8-gray_image.png')
inverted = util.invert(original)
intermediate_height = 300
scale_ratio = intermediate_height / inverted.shape[0]
resized_image = transform.resize(
    inverted,
    output_shape=(
        intermediate_height,
        int(inverted.shape[1] * scale_ratio)
    )
)
images.append(('Original', resized_image))

blurred = filters.gaussian(resized_image, sigma=1)
images.append(('blurred', blurred))

sobel_v = filters.sobel_v(blurred)
images.append(('sobel_v', sobel_v))

# sobel_v = filters.sobel_v(blurred)
# images.append(('sobel_v', sobel_v))

the_gradient = filters.rank.tophat(sobel_v, morphology.disk(10))
images.append(('gradient', the_gradient))

# images.append(('threshold_adaptive', filters.threshold_adaptive(edge_image)))

# images.append(('Hough Lines', sobel_v))
# lines = transform.probabilistic_hough_line(
#     sobel_v,
#     # threshold=10,
#     # line_length=resized_image.shape[0],
#     # line_gap=30,
#     # theta=numpy.array([math.tau/12, math.tau * 11/12]),
# )
# print(len(lines))

grid_width = int(math.sqrt(len(images)))
fig, axes = pyplot.subplots(
    nrows=math.ceil(len(images) / grid_width),
    ncols=grid_width,
    figsize=(8, 8),
    sharex=True, sharey=True
)
ax = axes.ravel()

for index, (title, image) in enumerate(images):
    ax[index].imshow(image, cmap=pyplot.cm.gray)
    ax[index].set_title(title)
    ax[index].axis('off')

# for line in lines[:100]:
#     p0, p1 = line
#     ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))

# fig.tight_layout()
pyplot.show()
