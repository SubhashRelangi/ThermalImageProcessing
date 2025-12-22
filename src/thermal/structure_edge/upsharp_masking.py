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
    Apply Unsharp Masking (USM) to enhance spatial sharpness in a
    single-channel image.

    Unsharp Masking works by subtracting a blurred (low-pass) version
    of the image from the original to extract high-frequency detail,
    then amplifying and adding it back to the original image.

    This implementation is designed for:
        - Thermal imagery
        - Grayscale preprocessing pipelines
        - ML-ready feature enhancement
        - Controlled sharpness amplification with noise awareness

    ---------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------

    image : str | np.ndarray
        Description:
            Input image to be sharpened. Can be either a file path
            or a NumPy array. The image must be single-channel
            (grayscale).

        Min & Max values:
            - Shape: (H, W)
            - Dtype:
                uint8   : [0, 255]
                uint16  : [0, 65535]
                float32 : [0.0, 1.0] (recommended)

        Units:
            Pixel intensity

        Default:
            Required

        Best-case values:
            Noise-reduced grayscale image with preserved edges.

    ---------------------------------------------------------------------

    blur_ksize : Tuple[int, int]
        Description:
            Kernel size for Gaussian blurring used to generate the
            low-frequency component.

        Min & Max values:
            - Each value must be positive and odd
            - Typical range: (3, 3) to (11, 11)

        Units:
            Pixels

        Default:
            (5, 5)

        Best-case values:
            (3, 3) for subtle sharpening,
            (5, 5) for general-purpose enhancement.

    ---------------------------------------------------------------------

    blur_sigma_x : float
        Description:
            Standard deviation of the Gaussian kernel in the X direction.
            If set to 0, OpenCV automatically derives it from kernel size.

        Min & Max values:
            ≥ 0.0

        Units:
            Pixels

        Default:
            0.0

        Best-case values:
            0.0 (automatic) for most use cases.

    ---------------------------------------------------------------------

    blur_sigma_y : float
        Description:
            Standard deviation of the Gaussian kernel in the Y direction.
            If set to 0, it defaults to the value of blur_sigma_x.

        Min & Max values:
            ≥ 0.0

        Units:
            Pixels

        Default:
            0.0

        Best-case values:
            0.0 for isotropic blurring.

    ---------------------------------------------------------------------

    mask_scale : float
        Description:
            Scaling factor applied to the high-frequency mask
            (original − blurred).

        Min & Max values:
            ≥ 0.0

        Units:
            Unitless gain

        Default:
            0.5

        Best-case values:
            0.3 – 0.7 for thermal and low-noise images.

    ---------------------------------------------------------------------

    sharp_alpha : float
        Description:
            Strength of sharpness enhancement applied when adding
            the high-frequency mask back to the original image.

        Min & Max values:
            ≥ 0.0

        Units:
            Unitless amplification factor

        Default:
            1.2

        Best-case values:
            1.0 – 1.5 for controlled enhancement without ringing.

    ---------------------------------------------------------------------

    out_min : float
        Description:
            Minimum allowed output pixel value after sharpening.

        Min & Max values:
            Depends on output dtype

        Units:
            Pixel intensity

        Default:
            0.0

        Best-case values:
            0.0 for standard grayscale output.

    ---------------------------------------------------------------------

    out_max : float
        Description:
            Maximum allowed output pixel value after sharpening.

        Min & Max values:
            Depends on output dtype

        Units:
            Pixel intensity

        Default:
            255.0

        Best-case values:
            255.0 for 8-bit visualization.

    ---------------------------------------------------------------------

    output_dtype : str | np.dtype | None
        Description:
            Desired output data type of the sharpened image.

        Allowed values:
            - np.uint8
            - np.uint16
            - np.float32 / np.float64
            - None (returns float32 result without conversion)

        Units:
            Pixel intensity representation

        Default:
            np.uint8

        Best-case values:
            - uint8  : visualization
            - float32: ML pipelines
            - uint16 : radiometric preservation

    ---------------------------------------------------------------------
    RETURNS
    ---------------------------------------------------------------------

    out : np.ndarray
        Description:
            Sharpened image after unsharp masking.

        Shape:
            (H, W)

        Dtype:
            Determined by output_dtype

        Units:
            Pixel intensity

    ---------------------------------------------------------------------
    EXCEPTIONS
    ---------------------------------------------------------------------

    Raises:
        - FileNotFoundError : If image path cannot be read
        - TypeError        : Invalid input types or dtypes
        - ValueError       : Invalid parameters or image shape

    ---------------------------------------------------------------------
    USE CASES
    ---------------------------------------------------------------------

    - Edge enhancement before segmentation
    - Thermal structure amplification
    - Feature sharpening for ML models
    - Preprocessing for gradient and edge operators

    ---------------------------------------------------------------------
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

