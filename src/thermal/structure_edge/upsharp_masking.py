import cv2 as cv
import numpy as np
import time
from typing import Tuple, Union, Optional, Dict

ImageArray = np.ndarray

def apply_unsharp_masking(
    image: Union[str, ImageArray],
    *,
    blur_ksize: Tuple[int, int] = (5, 5),
    blur_sigma_x: float = 0.0,
    blur_sigma_y: float = 0.0,
    mask_scale: float = 0.5,
    sharp_alpha: float = 1.2,
    out_min: float = 0.0,
    out_max: float = 255.0,
    output_dtype: Optional[Union[str, np.dtype]] = np.uint8,
) -> ImageArray:
    """
    Apply Unsharp Masking (USM) to enhance image sharpness.
    """

    start = time.perf_counter()

    # -------------------------------------------------
    # INPUT LOAD + VALIDATION (INLINE)
    # -------------------------------------------------
    if isinstance(image, str):
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError("image must be a file path or numpy array")

    if img.ndim == 3:
        img = img[..., 0]

    if img.ndim != 2 or img.size == 0:
        raise ValueError("Input image must be non-empty single-channel")

    if not isinstance(blur_ksize, tuple) or len(blur_ksize) != 2:
        raise TypeError("blur_ksize must be tuple(int, int)")

    kx, ky = int(blur_ksize[0]), int(blur_ksize[1])
    if kx <= 0 or ky <= 0:
        raise ValueError("Kernel sizes must be positive")
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    blur_ksize = (kx, ky)

    if sharp_alpha < 0:
        raise ValueError("sharp_alpha must be >= 0")
    if mask_scale < 0:
        raise ValueError("mask_scale must be >= 0")

    # -------------------------------------------------
    # TYPE NORMALIZATION → float32 [0,255]
    # -------------------------------------------------
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32)
    elif img.dtype == np.uint16:
        img_f = (img.astype(np.float32) / 65535.0) * 255.0
    elif np.issubdtype(img.dtype, np.floating):
        img_f = np.clip(img, 0.0, 1.0).astype(np.float32) * 255.0
    else:
        raise TypeError(f"Unsupported dtype: {img.dtype}")

    # -------------------------------------------------
    # GAUSSIAN BLUR
    # -------------------------------------------------
    blurred = cv.GaussianBlur(img_f, blur_ksize, blur_sigma_x, blur_sigma_y)

    # -------------------------------------------------
    # UNSHARP MASK
    # -------------------------------------------------
    mask = (img_f - blurred) * mask_scale
    sharpened = img_f + sharp_alpha * mask
    sharpened = np.clip(sharpened, out_min, out_max)

    # -------------------------------------------------
    # OUTPUT TYPE
    # -------------------------------------------------
    if output_dtype == np.uint8:
        out = sharpened.astype(np.uint8)
    elif output_dtype == np.uint16:
        out = (sharpened * 257).astype(np.uint16)
    elif output_dtype is None:
        out = sharpened
    else:
        out = sharpened.astype(output_dtype)

    print(f"[USM] Time: {time.perf_counter() - start:.4f}s")
    return out

