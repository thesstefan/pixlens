import numpy as np
from numpy.typing import NDArray


def apply_mask(np_image: NDArray, mask: NDArray) -> NDArray:
    # Ensure the mask is a boolean array
    mask = mask.astype(bool)

    # Apply the mask to each channel
    masked_image = np.zeros_like(np_image)
    for i in range(
        np_image.shape[2],
    ):  # Assuming image has shape [Height, Width, Channels]
        masked_image[:, :, i] = np_image[:, :, i] * mask

    return masked_image
