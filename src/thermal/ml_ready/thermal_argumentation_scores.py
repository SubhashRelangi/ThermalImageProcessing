import numpy as np
import cv2


def augmentation_difference_score(
    original: np.ndarray,
    augmented: np.ndarray
) -> dict:
    """
    Generic augmentation quality metrics.
    """

    if original.shape != augmented.shape:
        raise ValueError("Shape mismatch")

    diff = augmented.astype(np.float32) - original.astype(np.float32)

    return {
        "mean_difference": float(np.mean(diff)),
        "std_difference": float(np.std(diff)),
        "energy_ratio": float(
            np.mean(augmented.astype(np.float32) ** 2) /
            (np.mean(original.astype(np.float32) ** 2) + 1e-6)
        )
    }
