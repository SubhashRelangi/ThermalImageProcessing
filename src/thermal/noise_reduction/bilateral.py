import cv2
import numpy as np
import time


def thermal_bilateral_filter(
    src: np.ndarray,
    diameter: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 25.0
) -> np.ndarray:
    """
    Apply bilateral filtering to a single-channel thermal image.

    This function performs edge-preserving noise reduction by combining
    spatial proximity and intensity similarity, making it suitable for
    thermal imagery where temperature discontinuities must be preserved.

    Parameters
    ----------
    src : np.ndarray
        Input single-channel thermal image.
        Shape: (H, W) or (H, W, 1)

    diameter : int, optional
        Diameter of pixel neighborhood used for filtering.
        Default is 5.

    sigma_color : float, optional
        Filter sigma in the intensity (thermal value) domain.
        Default is 25.0.

    sigma_space : float, optional
        Filter sigma in the spatial domain.
        Default is 25.0.

    Returns
    -------
    np.ndarray
        Bilaterally filtered thermal image with the same shape as input.

    Raises
    ------
    ValueError
        If input is None, not single-channel, or parameters are invalid.

    TypeError
        If input is not a numpy array.
    """

    start = time.time()

    if src is None:
        raise ValueError("Source image is None")

    if not isinstance(src, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")

    # --- Strict thermal rule: single-channel only ---
    if src.ndim == 2:
        img = src
    elif src.ndim == 3 and src.shape[2] == 1:
        img = src[..., 0]
    else:
        raise ValueError(
            f"Single-channel thermal image required. Got shape={src.shape}"
        )

    if diameter <= 0:
        raise ValueError("diameter must be positive")

    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("sigma_color and sigma_space must be > 0")

    out_img = cv2.bilateralFilter(
        img,
        diameter,
        sigma_color,
        sigma_space
    )

    end = time.time()

    print(f"[thermal_median_blur] Time: {end - start:.6f}s")

    return out_img
