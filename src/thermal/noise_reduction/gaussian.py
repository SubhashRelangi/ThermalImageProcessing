import cv2
import numpy as np
from typing import Optional, Tuple, Union
import time

ImageArray = np.ndarray


def thermal_gaussian_filter(
    image: ImageArray,
    *,
    ksize: Tuple[int, int] = (5, 5),
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
    output_dtype: Optional[Union[str, np.dtype]] = None,
) -> ImageArray:
    """
    Apply Gaussian smoothing to a single-channel thermal image.
    """

    start = time.time()

    if image is None:
        raise ValueError("Input image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")

    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    elif image.ndim != 2:
        raise ValueError(
            f"Single-channel thermal image required. Got shape={image.shape}"
        )

    # ---- dtype normalization ----
    if image.dtype == np.uint8:
        img_f = image.astype(np.float32)
    elif image.dtype == np.uint16:
        img_f = (image.astype(np.float32) / 65535.0) * 255.0
    elif np.issubdtype(image.dtype, np.floating):
        img_f = np.clip(image, 0.0, 1.0).astype(np.float32) * 255.0
    else:
        raise TypeError(f"Unsupported dtype {image.dtype}")

    # ---- kernel validation ----
    if not (isinstance(ksize, tuple) and len(ksize) == 2):
        raise TypeError("ksize must be a tuple of two ints")

    kx, ky = map(int, ksize)
    if kx <= 0 or ky <= 0:
        raise ValueError("Kernel size must be positive")

    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1

    filtered = cv2.GaussianBlur(
        img_f,
        (kx, ky),
        sigmaX=float(sigma_x),
        sigmaY=float(sigma_y),
    )

    # ---- output dtype ----
    if output_dtype is None or np.dtype(output_dtype) == np.uint8:
        return np.clip(filtered, 0, 255).astype(np.uint8)

    if np.dtype(output_dtype) == np.uint16:
        return (np.clip(filtered, 0, 255).astype(np.uint16) * 257)

    out_img = filtered.astype(output_dtype)

    end = time.time()

    print(f"[thermal_median_blur] Time: {end - start:.6f}s")
    
    return out_img