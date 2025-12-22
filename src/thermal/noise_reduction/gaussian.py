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
    Apply Gaussian Blur to a single-channel thermal image.

    This function performs spatial Gaussian smoothing, commonly used in
    thermal image processing to reduce sensor noise and suppress
    high-frequency artifacts while preserving low-frequency thermal structure.

    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------

    image : str | np.ndarray
        Description:
            Input thermal image. Can be either a file path or an already loaded
            image array. The image must be single-channel (grayscale).

        Min & Max values:
            - Pixel values depend on dtype:
              * uint8  : 0 – 255
              * uint16 : 0 – 65535
              * float  : expected in [0.0 – 1.0]

        Units:
            Pixel intensity (radiometric proxy)

        Default value:
            None

        Best-case values:
            Radiometrically calibrated, noise-present thermal image
            with correct dynamic range.

    --------------------------------------------------------------------------

    ksize : Tuple[int, int]
        Description:
            Gaussian kernel size in (width, height). Kernel dimensions are
            automatically enforced to be positive and odd.

        Min & Max values:
            - Minimum: (1, 1)
            - Maximum: Practically limited by image size (H, W)

        Units:
            Pixels

        Default value:
            (5, 5)

        Best-case values:
            (3, 3) or (5, 5) for most thermal denoising tasks.

    --------------------------------------------------------------------------

    sigma_x : float
        Description:
            Standard deviation of the Gaussian kernel along the X-axis.
            Controls horizontal smoothing strength. If set to 0, OpenCV
            computes sigma from kernel size.

        Min & Max values:
            - Minimum: 0.0
            - Maximum: No hard limit (practically ≤ 5.0)

        Units:
            Pixels

        Default value:
            0.0

        Best-case values:
            1.0 – 2.0 for thermal noise reduction.

    --------------------------------------------------------------------------

    sigma_y : float
        Description:
            Standard deviation of the Gaussian kernel along the Y-axis.
            Controls vertical smoothing strength. If set to 0, it defaults
            to sigma_x.

        Min & Max values:
            - Minimum: 0.0
            - Maximum: No hard limit (practically ≤ 5.0)

        Units:
            Pixels

        Default value:
            0.0

        Best-case values:
            Same as sigma_x (typically 1.0 – 2.0).

    --------------------------------------------------------------------------

    output_dtype : str | np.dtype | None
        Description:
            Desired output data type. If None, output is returned as uint8.
            Used mainly for visualization or downstream compatibility.

        Min & Max values:
            Supported types:
                - np.uint8
                - np.uint16
                - np.float32 / np.float64

        Units:
            Pixel intensity

        Default value:
            None (defaults to uint8)

        Best-case values:
            - uint8  : visualization
            - float32: further thermal processing
            - uint16 : radiometric preservation

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------

    output : np.ndarray
        Description:
            Gaussian-blurred thermal image with the specified output dtype.

        Units:
            Pixel intensity

    --------------------------------------------------------------------------
    Notes
    --------------------------------------------------------------------------
    - Only single-channel images are supported.
    - 3-channel (RGB/BGR) inputs will raise an exception.
    - Internal processing is performed in float32 for numerical stability.
    - This function does NOT perform radiometric calibration.

    --------------------------------------------------------------------------
    Typical Thermal Use Case
    --------------------------------------------------------------------------
    - Preprocessing before edge detection
    - Noise suppression prior to segmentation
    - Temporal stabilization pipelines

    --------------------------------------------------------------------------
    """

    start = time.time()

    try:
        # --------------------------------------------------
        # Input validation
        # --------------------------------------------------
        if image is None:
            raise ValueError("Input image is None")

        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy ndarray")

        # --------------------------------------------------
        # Channel validation
        # --------------------------------------------------
        if image.ndim == 3 and image.shape[2] == 1:
            img = image[..., 0]
        elif image.ndim == 2:
            img = image
        else:
            raise ValueError(
                f"Single-channel thermal image required. Got shape={image.shape}"
            )

        # --------------------------------------------------
        # Dtype normalization → float32 [0, 255]
        # --------------------------------------------------
        if img.dtype == np.uint8:
            img_f = img.astype(np.float32)

        elif img.dtype == np.uint16:
            img_f = (img.astype(np.float32) / 65535.0) * 255.0

        elif np.issubdtype(img.dtype, np.floating):
            img_f = np.clip(img, 0.0, 1.0).astype(np.float32) * 255.0

        else:
            raise TypeError(f"Unsupported dtype {img.dtype}")

        # --------------------------------------------------
        # Kernel validation
        # --------------------------------------------------
        if not (isinstance(ksize, tuple) and len(ksize) == 2):
            raise TypeError("ksize must be a tuple of two integers")

        kx, ky = int(ksize[0]), int(ksize[1])

        if kx <= 0 or ky <= 0:
            raise ValueError("Kernel size must be positive")

        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1

        # --------------------------------------------------
        # Gaussian filtering
        # --------------------------------------------------
        filtered = cv2.GaussianBlur(
            img_f,
            (kx, ky),
            sigmaX=float(sigma_x),
            sigmaY=float(sigma_y),
        )

        # --------------------------------------------------
        # Output dtype handling
        # --------------------------------------------------
        if output_dtype is None or np.dtype(output_dtype) == np.uint8:
            return np.clip(filtered, 0, 255).astype(np.uint8)

        if np.dtype(output_dtype) == np.uint16:
            return (np.clip(filtered, 0, 255).astype(np.uint16) * 257)

        return filtered.astype(output_dtype)

    except Exception as e:
        # Centralized error tagging
        raise RuntimeError(f"[thermal_gaussian_filter] {e}") from e

    finally:
        end = time.time()
        print(f"[thermal_gaussian_filter] Time: {end - start:.6f}s")
