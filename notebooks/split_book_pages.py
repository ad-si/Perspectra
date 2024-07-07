import marimo

__generated_with = "0.7.0"
app = marimo.App()


@app.cell
def __():
    import os
    from typing import List, Any
    import numpy as np
    import skimage as skimage
    from skimage import io, transform, filters
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    return Any, List, filters, io, norm, np, os, plt, skimage, transform


@app.cell
def __(__file__, io, os, skimage, transform):
    def showImg(image):
        return io.imshow(arr=image, plugin="matplotlib")


    input_image_path = os.path.join(
        os.path.dirname(__file__), "../tests/fixtures/book_color.jpeg"
    )
    book = io.imread(fname=input_image_path)
    book_grayscale = skimage.color.rgb2gray(book)
    resized = transform.downscale_local_mean(book_grayscale, (4, 4))
    showImg(resized)
    return book, book_grayscale, input_image_path, resized, showImg


@app.cell
def __(norm, np, plt, resized):
    img_width = resized.shape[1]
    img_height = resized.shape[0]


    def get_sine_probability(img_width, img_height):
        samples = np.arange(0, np.pi, np.pi / img_width)
        amplitude = np.sin(samples)
        gradient_image = np.broadcast_to(amplitude, (img_height, img_width))
        probabilitized = resized * gradient_image
        return probabilitized


    def get_norm_probability(img_width):
        samples = np.arange(img_width)
        spread = 10
        amplitude = norm.pdf(samples, img_width / 2, img_width / spread)
        amplitude *= 1 / amplitude.max()
        return amplitude


    def apply_probability(probability, image):
        gradient_image = np.broadcast_to(
            probability, (image.shape[0], image.shape[1])
        )
        probabilitized = image * gradient_image
        return probabilitized


    pxInInch = 0.008
    prob_norm = get_norm_probability(img_width)
    image_prob = apply_probability(prob_norm, resized)
    fig, items = plt.subplots(
        ncols=2, figsize=(3 * img_width * pxInInch, img_height * pxInInch)
    )
    items[0].plot(np.arange(img_width), prob_norm)
    items[0].set_title("Probability function")
    items[0].grid(True, which="both")
    items[0].axhline(y=0, color="k")
    items[1].set_title("Idealized probability for book fold")
    items[1].imshow(image_prob)
    fig.tight_layout()
    plt.show()
    return (
        apply_probability,
        fig,
        get_norm_probability,
        get_sine_probability,
        image_prob,
        img_height,
        img_width,
        items,
        prob_norm,
        pxInInch,
    )


@app.cell
def __(apply_probability, filters, io, prob_norm, resized):
    sobeled = filters.sobel_v(image=resized)
    io.imshow(apply_probability(prob_norm, sobeled))
    return sobeled,


@app.cell
def __(List, input_image_path):
    import imageio.v3 as imageio


    def split_book(image) -> List[List[int]]:
        return []


    image = imageio.imread(input_image_path, rotate=True)
    pages = split_book(image)
    pages
    return image, imageio, pages, split_book


if __name__ == "__main__":
    app.run()
