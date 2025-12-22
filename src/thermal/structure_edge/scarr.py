import cv2 as cv
import numpy as np
import sys
import time
from typing import Optional, Dict

Image = np.ndarray

def apply_scharr_operator(
    image: Image,
    *,
    blur_ksize=3,
    threshold_val=45.0,
    scharr_ddepth=cv.CV_32F,
    dx_scharr_x=1,
    dy_scharr_x=0,
    dx_scharr_y=0,
    dy_scharr_y=1,
    threshold_max_value=255.0,
    threshold_type=cv.THRESH_BINARY
) -> Image:
    """
    Apply Scharr edge detection with thermal-safe preprocessing.

    Parameters
    ----------
    image : ndarray
        Description:
            Input single-channel thermal image.
        Min & Max:
            Any valid grayscale range (commonly 0–255).
        Units:
            Intensity.
        Default:
            Required.
        Best case:
            8-bit or 16-bit thermal frame, noise-reduced.

    blur_ksize : int
        Description:
            Gaussian blur kernel size to suppress sensor noise.
        Min & Max:
            Min = 1, Max = 31 (odd only).
        Units:
            Pixels.
        Default:
            5
        Best case:
            3–7 for thermal images.

    threshold_val : float
        Description:
            Threshold on normalized gradient magnitude.
        Min & Max:
            0–255.
        Units:
            Intensity.
        Default:
            60
        Best case:
            40–80 depending on noise.

    scharr_ddepth : int
        Description:
            Output depth of Scharr gradients.
        Default:
            cv.CV_32F
        Best case:
            CV_32F for precision.

    Returns
    -------
    edge_map : ndarray
        Binary Scharr edge map.
    """

    try:
        start_time = time.time()

        if image is None:
            raise ValueError("Input image is None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy ndarray.")
        if image.ndim != 2:
            raise ValueError("Input must be single-channel.")
        if blur_ksize < 1 or blur_ksize % 2 == 0:
            raise ValueError("blur_ksize must be odd and >= 1.")

        # --- Pre-blur ---
        image_blur = cv.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

        # --- Scharr gradients ---
        gx = cv.Scharr(image_blur, scharr_ddepth, dx_scharr_x, dy_scharr_x)
        gy = cv.Scharr(image_blur, scharr_ddepth, dx_scharr_y, dy_scharr_y)

        # --- Gradient magnitude ---
        grad_mag = cv.magnitude(gx, gy)

        # --- Normalize ---
        mag_norm = cv.normalize(
            grad_mag, None, 0.0, 255, cv.NORM_MINMAX
        )

        mag_u8 = np.clip(mag_norm, 0, 255).astype(np.uint8)

        # --- Threshold ---
        _, edge_map = cv.threshold(
            mag_u8, threshold_val, threshold_max_value, threshold_type
        )

        print(f"[Scharr] Time: {time.time() - start_time:.6f} sec")
        return edge_map

    except Exception as e:
        raise RuntimeError(f"Scharr operator failed: {e}") from e

