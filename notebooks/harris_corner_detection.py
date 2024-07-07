import marimo

__generated_with = "0.7.0"
app = marimo.App()


@app.cell
def __():
    import os
    import logging

    import imageio.v3 as imageio
    from skimage.feature import corner_harris, corner_peaks
    from skimage.util import img_as_ubyte
    from skimage import exposure


    output_base_path = "output"
    debug = True


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
            if self.level != "debug":
                return
            self.step_counter += 1
            image_path = os.path.join(
                self.base_path,
                f"{self.step_counter}-{name}.png",
            )
            imageio.imwrite(image_path, image)
            logging.info(f"Stored image: {image_path}")
            return self


    debugger = ImageDebugger(
        level="debug" if debug else "",
        base_path=output_base_path,
    )


    def get_harris_peaks(image, sigma, k):
        img_harris = corner_harris(image, sigma=sigma, k=k)
        debugger.save(
            "harris_corner_response",
            img_as_ubyte(
                exposure.rescale_intensity(
                    img_harris,
                ),
            ),
        )

        peaks_image = corner_peaks(
            img_harris,
            min_distance=5,  # Prevent inclusion of `image_corners`
            indices=False,
        )
        debugger.save("harris_corner_peaks", peaks_image)

        peaks = corner_peaks(
            img_harris,
            min_distance=5,
        )

        return peaks
    return (
        ImageDebugger,
        corner_harris,
        corner_peaks,
        debug,
        debugger,
        exposure,
        get_harris_peaks,
        imageio,
        img_as_ubyte,
        logging,
        os,
        output_base_path,
    )


if __name__ == "__main__":
    app.run()
