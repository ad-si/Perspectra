import marimo

__generated_with = "0.7.0"
app = marimo.App()


@app.cell
def __():
    import numpy as np
    import scipy.ndimage.filters as filters
    from skimage import data
    import matplotlib.pyplot as plt


    def add_border_approx_color(arr, border_size=1):
        # Compute the local mean color of each pixel
        local_mean = filters.uniform_filter(
            arr, size=border_size * 2 + 1, mode="reflect"
        )

        # Compute the difference between each pixel and its local mean
        diff = np.abs(arr - local_mean)

        # Scale the difference so that it ranges from 0 to 1
        diff /= diff.max()

        # Create a mask for the border
        mask = np.zeros(arr.shape)
        mask[border_size:-border_size, border_size:-border_size] = 1
        mask = filters.gaussian_filter(mask, border_size)

        # Apply the mask to the difference array to get the border color
        border_color = np.zeros_like(arr)
        for i in range(arr.shape[-1]):
            border_color[..., i] = np.mean(diff[..., i] * mask)

        # Create a new array with the border color
        border_arr = np.zeros(
            (
                arr.shape[0] + border_size * 2,
                arr.shape[1] + border_size * 2,
                arr.shape[2],
            )
        )
        border_arr[border_size:-border_size, border_size:-border_size] = arr
        border_arr[:border_size, :, :] = border_color[:border_size, :, :]
        border_arr[-border_size:, :, :] = border_color[-border_size:, :, :]
        border_arr[:, :border_size, :] = border_color[:, :border_size, :]
        border_arr[:, -border_size:, :] = border_color[:, -border_size:, :]

        return border_arr
    return add_border_approx_color, data, filters, np, plt


@app.cell
def __(data, plt):
    plt.imshow(data.camera(), cmap="gray")
    plt.axis("off")
    plt.show()
    return


@app.cell
def __(add_border_approx_color, data, plt):
    image = add_border_approx_color(data.camera())

    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()
    return image,


if __name__ == "__main__":
    app.run()
