import io

import matplotlib.pyplot as plt
from PIL import Image


def figure_to_image(figure: plt.Figure) -> Image.Image:
    buffer = io.BytesIO()
    figure.savefig(buffer)
    plt.close(figure)
    buffer.seek(0)

    return Image.open(buffer)
