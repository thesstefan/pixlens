import numpy as np
import torch
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


def apply_segmentation_mask(
    image: Image.Image,
    mask: torch.Tensor,
) -> Image.Image:
    """Apply a segmentation mask to an image, coloring the masked parts black.

    :param image: PIL.Image - The original image
    :param mask: np.array - A boolean array representing the segmentation mask
    :return: PIL.Image - The image with the segmentation applied
    """
    # Convert the image to a numpy array
    img_array = np.array(image)

    # Ensure the mask is a boolean array
    mask = mask.cpu().numpy().astype(bool)

    # Check if dimensions of the mask and the image match
    if img_array.shape[:2] != mask.shape[-2:]:
        raise ValueError(
            "The dimensions of the image and the mask do not match."
        )

    # Apply the mask: set pixels to black where mask is True
    img_array[mask] = [0, 0, 0]

    # Convert the array back to an image
    return Image.fromarray(img_array)


def compute_union_segmentation_masks(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError


def masked_image_similarity(
    image1: Image.Image,
    mask1: torch.Tensor,
    image2: Image.Image,
    mask2: torch.Tensor,
) -> float:
    new_mask = compute_union_segmentation_masks(mask1, mask2)
    masked_image1 = apply_segmentation_mask(image1, new_mask)
    masked_image2 = apply_segmentation_mask(image2, new_mask)
    return image_similarity(masked_image1, masked_image2)

