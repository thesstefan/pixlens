import enum

import numpy as np
import numpy.typing as npt
import torch

from pixlens.evaluation.utils import center_of_mass


class MultiplicityResolution(enum.StrEnum):
    LARGEST = "LARGEST"
    MOST_CONFIDENT = "MOST_CONFIDENT"
    CLOSEST = "CLOSEST"


def select_largest_2d(masks_2d: npt.NDArray) -> int:
    assert len(masks_2d.shape) in [3, 4]  # noqa: S101

    if len(masks_2d.shape) == 4:  # noqa: PLR2004
        masks_2d = np.squeeze(masks_2d, axis=1)

    return np.argmax(np.count_nonzero(masks_2d, axis=(-2, -1))).item()


def select_closest_2d(masks_2d: npt.NDArray, relative: npt.NDArray) -> int:
    # TODO: Update center_of_mass to use np arrays and not depend on shape
    # TODO: Make this take a relative point instead

    assert len(masks_2d.shape) in [3, 4]  # noqa: S101
    assert len(relative.shape) in [2, 3]  # noqa: S101

    if len(masks_2d.shape) == 3:  # noqa: PLR2004
        masks_2d = np.expand_dims(masks_2d, 1)

    if len(relative.shape) == 2:  # noqa: PLR2004
        relative = np.expand_dims(relative, 0)

    relative_center = np.array(center_of_mass(torch.Tensor(relative)))
    mask_centers = [
        np.array(center_of_mass(torch.Tensor(mask))) for mask in masks_2d
    ]

    return np.argmin(
        [
            np.linalg.norm(mask_center - relative_center)
            for mask_center in mask_centers
        ],
    ).item()


# Kind of stupid, but allows us to be certain that the same thing
# is done for all operations.
def select_one_2d(
    masks_2d: npt.NDArray,
    resolution: MultiplicityResolution,
    confidences: npt.NDArray | None = None,
    relative_mask: npt.NDArray | None = None,
) -> int:
    match resolution:
        case MultiplicityResolution.LARGEST:
            return select_largest_2d(masks_2d)
        case MultiplicityResolution.MOST_CONFIDENT:
            assert confidences is not None  # noqa: S101
            assert len(masks_2d) == len(confidences)  # noqa: S101
            assert len(confidences.shape) == 1  # noqa: S101

            return np.argmax(confidences).item()
        case MultiplicityResolution.CLOSEST:
            assert relative_mask is not None  # noqa: S101

            return select_closest_2d(masks_2d, relative_mask)
        case _:
            msg = f"{resolution} is not a supported resolution type!"
            raise RuntimeError(msg)
