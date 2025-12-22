import cv2
import numpy as np
import time


def thermal_median_filter(
    src: np.ndarray,
    ksize: int = 3
) -> np.ndarray:
    """
    Apply median filtering to a single-channel thermal image.
    Suppresses impulsive noise while preserving edges.
    """

    start = time.time()

    if src is None:
        raise ValueError("Input image is None")

    if not isinstance(src, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")

    if src.size == 0:
        raise ValueError("Input image is empty")

    if src.ndim == 3 and src.shape[2] == 1:
        img = src[..., 0]
    elif src.ndim == 2:
        img = src
    else:
        raise ValueError("Median filter requires single-channel image")

    if not isinstance(ksize, int):
        raise TypeError("ksize must be an integer")

    if ksize <= 1 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer > 1")

    out_img = cv2.medianBlur(img, ksize)

    end = time.time()

    print(f"[thermal_median_blur] Time: {end - start:.6f}s")

    return out_img
