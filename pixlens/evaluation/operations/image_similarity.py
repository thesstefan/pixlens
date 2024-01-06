import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error


def image_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute the similarity between two PIL images.

    img1: Ideally original PIL image
    img2: Ideally edited PIL image (this one gets resized to the shape of
    first image if needed)
    """
    # Resize the second image to match the size of the first image if needed
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    mse_value: float = mean_squared_error(
        img1_array.flatten(), img2_array.flatten()
    )
    ssim_value: float = ssim(img1_array, img2_array, multichannel=True)

    return (
        1 - mse_value
    ) * ssim_value  # (We can modify this as we want honestly)
