import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_color_histograms(
    bgr_color_histograms: npt.NDArray[np.uint],
) -> plt.Figure:
    assert len(bgr_color_histograms.shape) == 2  # noqa: PLR2004, S101

    fig, axes = plt.subplots(
        bgr_color_histograms.shape[0],
        3,
        figsize=(20, 15),
        sharey=True,
    )

    bin_count = bgr_color_histograms.shape[1] // 3
    bin_bar_width = bin_count / 8
    x_values = [255 / bin_count * i for i in range(bin_count)]

    for i, histogram in enumerate(bgr_color_histograms):
        # histogram is 1x(C * bins) BGR
        blue_histogram, green_histogram, red_histogram = np.split(histogram, 3)

        axes[i, 0].bar(
            x_values,
            blue_histogram,
            width=bin_bar_width,
            color="b",
        )
        axes[i, 1].bar(
            x_values,
            green_histogram,
            width=bin_bar_width,
            color="g",
        )
        axes[i, 2].bar(
            x_values,
            red_histogram,
            width=bin_bar_width,
            color="r",
        )

        def describe_ax(ax: plt.Axes, color: str) -> None:
            ax.set_xlabel("Color Bins")
            ax.set_ylabel("Bin Color Occurences / Image Pixel Count")
            ax.set_title(f"Color Histogram ({color}, bins={bin_count})")

        describe_ax(axes[i, 0], "BLUE")
        describe_ax(axes[i, 1], "GREEN")
        describe_ax(axes[i, 2], "RED")

    return fig
