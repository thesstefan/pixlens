import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_rgb_histograms(
    rgb_color_histograms: npt.NDArray[np.uint],
    bin_count: int = 256,
) -> plt.Figure:
    assert len(rgb_color_histograms.shape) == 2  # noqa: PLR2004, S101
    assert 256 % (rgb_color_histograms.shape[1] / 3) == 0  # noqa: S101

    fig, axes = plt.subplots(
        rgb_color_histograms.shape[0],
        3,
        figsize=(20, 15),
        sharey=True,
    )

    bin_count = rgb_color_histograms.shape[1] // 3
    values_per_bin = 256 // bin_count
    x_values = [values_per_bin * i for i in range(bin_count)]

    def describe_ax(ax: plt.Axes) -> None:
        ax.set_xlabel("Bins")
        ax.set_ylabel("# of Pixles / Image Pixel Count")
        ax.set_title(f"Color Histogram (bins={bin_count})")

    for i, histogram in enumerate(rgb_color_histograms):
        # histogram is 1x(C * bins) BGR
        red_histogram, green_histogram, blue_histogram = np.split(histogram, 3)

        axes[i, 0].plot(x_values, red_histogram, color="r")
        axes[i, 1].plot(x_values, green_histogram, color="g")
        axes[i, 2].plot(x_values, blue_histogram, color="b")

        describe_ax(axes[i, 0])
        describe_ax(axes[i, 1])
        describe_ax(axes[i, 2])

    return fig
