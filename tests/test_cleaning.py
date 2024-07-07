import os
import math
from skimage import io, util
import matplotlib
from matplotlib import pyplot
from perspectra.multipass_cleaner import remove_noise
from perspectra.noise_generator import add_noise

matplotlib.use("Agg")

img_path = os.path.join(os.path.dirname(__file__), "fixtures/doc_binary.png")
original = util.invert(io.imread(img_path))
images = []
noisy_orig = add_noise(original)
images.append(("Noisy Original", noisy_orig))

# Should actually go up to ~8, but performance becomes unbearable
cleaned_img = remove_noise(noisy_orig, images=images)
images.append(("Final cleaned image", cleaned_img))
images.append(("Original", original))

grid_width = int(math.sqrt(len(images)))
fig, axes = pyplot.subplots(
    nrows=math.ceil(len(images) / grid_width),
    ncols=grid_width,
    figsize=(8, 8),
    sharex=True,
    sharey=True,
)
ax = axes.ravel()

for index, (title, image) in enumerate(images):
    ax[index].imshow(image, cmap=pyplot.cm.gray)
    ax[index].set_title(title)
    ax[index].axis("off")

fig.tight_layout()
pyplot.savefig("tests/test_cleaning_out.png")
