import marimo

__generated_with = "0.7.0"
app = marimo.App()


@app.cell
def __(circle, numpy):
    import os
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
    from skimage.draw import disk, circle_perimeter

    import matplotlib.pyplot as plt
    import numpy as np


    def show_images(images, cols=1, titles=None):
        """
        Display a list of images in a single figure with matplotlib.

        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.

        cols (Default = 1): Number of columns in figure (number of rows is
                            set to np.ceil(n_images/float(cols))).

        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """

        assert (titles is None) or (len(images) == len(titles))
        n_images = len(images)
        if titles is None:
            titles = ["Image (%d)" % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        fig.set_size_inches(15, 15)
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(
                cols, round(np.ceil(n_images / float(cols))), n + 1
            )
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()


    def load_image(file_name):
        file_path = file_name
        #     file_path = os.path.join(os.getcwd(), file_name)
        return io.imread(file_path)


    def get_marked_image(image, corners):
        radius = image.size // 2**18
        circles = [circle(c, r, radius) for r, c in corners]
        for circ in circles:
            image[circ] = (255, 0, 0)
        return image


    def get_fixed_image(image, corners):
        rows = image.shape[0]
        colums = round(image.shape[1])
        src_corners = [
            (0, 0),
            (0, rows),
            (colums, rows),
            (colums, 0),
        ]

        protrans = transform.ProjectiveTransform()
        protrans.estimate(numpy.array(src_corners), numpy.array(corners))

        return transform.warp(image, protrans, output_shape=image.shape)


    image = load_image("tests/fixtures/doc_photo.jpeg")
    corners = [
        (100, 15),
        (20, 290),
        (505, 155),
        (410, 30),
    ]

    show_images([image, morphology.closing(color.rgb2gray(image))])
    return (
        circle_perimeter,
        color,
        corners,
        disk,
        draw,
        exposure,
        feature,
        filters,
        get_fixed_image,
        get_marked_image,
        image,
        io,
        load_image,
        measure,
        morphology,
        np,
        os,
        plt,
        segmentation,
        show_images,
        skimage,
        transform,
        util,
    )


if __name__ == "__main__":
    app.run()
