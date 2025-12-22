import cv2
import numpy as np
import time


def thermal_median_filter(
    src: np.ndarray,
    ksize: int = 3
) -> np.ndarray:
    """
    Apply median filtering to a single-channel thermal image to suppress
    impulsive noise (e.g., hot pixels) while preserving edges and
    temperature boundaries.

    ---------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------
    src : np.ndarray
        Description:
            Input thermal image (single-channel).

        Min & Max values:
            uint8  : [0, 255]
            uint16 : [0, 65535]
            float  : [0.0, 1.0]

        Units:
            Thermal intensity (relative or calibrated).

        Best-case values:
            Sparse impulsive noise with preserved gradients.

    ksize : int
        Description:
            Median kernel size (odd integer > 1).

        Min & Max values:
            Min: 3
            Max: < min(image height, width)

        Units:
            Pixels

        Default:
            3

        Best-case values:
            3 or 5 for most thermal sensors.

    ---------------------------------------------------------------------
    RETURNS
    ---------------------------------------------------------------------
    np.ndarray
        Median-filtered thermal image.

    ---------------------------------------------------------------------
    EXCEPTIONS
    ---------------------------------------------------------------------
    Raises RuntimeError on validation or OpenCV failure.
    """

    start = time.time()

    try:
        # -----------------------------
        # Input validation
        # -----------------------------
        if src is None:
            raise ValueError("Input image is None")

        if not isinstance(src, np.ndarray):
            raise TypeError("Input must be a numpy ndarray")

        if src.size == 0:
            raise ValueError("Input image is empty")

        # -----------------------------
        # Channel validation
        # -----------------------------
        if src.ndim == 3 and src.shape[2] == 1:
            img = src[..., 0]
        elif src.ndim == 2:
            img = src
        else:
            raise ValueError("Median filter requires single-channel image")

        # -----------------------------
        # Kernel validation
        # -----------------------------
        if not isinstance(ksize, int):
            raise TypeError("ksize must be an integer")

        if ksize <= 1 or ksize % 2 == 0:
            raise ValueError("ksize must be an odd integer > 1")

        # -----------------------------
        # Median filtering
        # -----------------------------
        out_img = cv2.medianBlur(img, ksize)

        return out_img

    except Exception as e:
        # Centralized, tagged error
        raise RuntimeError(f"[thermal_median_filter] {e}") from e

    finally:
        end = time.time()
        print(f"[thermal_median_filter] Time: {end - start:.6f}s")
