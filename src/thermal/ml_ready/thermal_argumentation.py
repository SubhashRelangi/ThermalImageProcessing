import cv2
import numpy as np
import random
from typing import Union


# =====================================================
# THERMAL FLIP
# =====================================================
def thermal_flip(
    image: Union[str, np.ndarray],
    *,
    flip_code: int = 1
) -> np.ndarray:
    
    """
    Flip a thermal (single-channel) image using OpenCV flip codes.

    PARAMETERS
    ----------
    image : str | np.ndarray
        Description:
            Input thermal image or file path.

        Min / Max:
            - Minimum size: 1 × 1 pixels
            - Maximum size: system memory bound

        Units:
            Pixel intensity

        Supported dtypes:
            uint8, uint16, float32, float64

        Default:
            Required (no default)

        Best-case:
            uint8 / uint16 single-channel thermal image

    flip_code : int
        Description:
            Specifies the flip direction.

        Allowed values:
            1   → Horizontal flip (left ↔ right)
            0   → Vertical flip (top ↔ bottom)
           -1   → Both axes (180° rotation)

        Units:
            Discrete code (OpenCV convention)

        Default:
            1

        Best-case:
            1 (horizontal) for ML augmentation

    RETURNS
    -------
    out_image : np.ndarray
        Flipped thermal image (same shape and dtype as input).

    EXCEPTIONS
    ----------
    ValueError:
        - Invalid flip_code
        - Input image is None / empty
        - Image is not 2D

    FileNotFoundError:
        - Image path cannot be read

    TypeError:
        - Unsupported dtype
    """

    if flip_code not in (-1, 0, 1):
        raise ValueError("flip_code must be -1, 0, or 1")

    # --- Load ---
    if image is None:
        raise ValueError("Input image is None")

    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError("image must be path or numpy array")

    # --- Validate ---
    if img.size == 0:
        raise ValueError("Input image is empty")

    if img.ndim == 3:
        img = img[..., 0]

    if img.ndim != 2:
        raise ValueError("Thermal image must be single-channel")

    if not (
        img.dtype == np.uint8
        or img.dtype == np.uint16
        or np.issubdtype(img.dtype, np.floating)
    ):
        raise TypeError(f"Unsupported dtype {img.dtype}")

    out_img = cv2.flip(img, flip_code)
    return out_img


# =====================================================
# THERMAL ROTATE
# =====================================================
def thermal_rotate(
    image: Union[str, np.ndarray],
    *,
    angle: float,
    scale: float = 1.0,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101
) -> np.ndarray:
    
    """
    Rotate a thermal (single-channel) image.

    PARAMETERS
    ----------
    image : str | np.ndarray
        Input thermal image or file path.

    angle : float
        Description:
            Rotation angle in degrees.
            Positive values rotate counter-clockwise.

        Min / Max:
            No hard limit (practically -360 to +360)

        Units:
            Degrees

        Default:
            Required (no default)

        Best-case:
            Small angles (±15°) for augmentation

    scale : float
        Description:
            Isotropic scaling factor applied during rotation.

        Min / Max:
            > 0

        Units:
            Unitless scale factor

        Default:
            1.0

        Best-case:
            1.0 (no scaling)

    interpolation : int
        Description:
            Interpolation method used by OpenCV during rotation.

        Allowed values:
            cv2.INTER_NEAREST
            cv2.INTER_LINEAR
            cv2.INTER_CUBIC
            cv2.INTER_AREA

        Units:
            OpenCV enum

        Default:
            cv2.INTER_LINEAR

        Best-case:
            cv2.INTER_LINEAR (balanced for thermal images)

    border_mode : int
        Description:
            Pixel extrapolation method for borders.

        Allowed values:
            cv2.BORDER_CONSTANT
            cv2.BORDER_REPLICATE
            cv2.BORDER_REFLECT
            cv2.BORDER_REFLECT_101

        Units:
            OpenCV enum

        Default:
            cv2.BORDER_REFLECT_101

        Best-case:
            cv2.BORDER_REFLECT_101 (avoids artificial cold borders)

    RETURNS
    -------
    out_image : np.ndarray
        Rotated thermal image (same shape and dtype as input).
    """

    if scale <= 0:
        raise ValueError("scale must be > 0")

    if image is None:
        raise ValueError("Input image is None")

    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError("image must be path or numpy array")

    if img.size == 0:
        raise ValueError("Input image is empty")

    if img.ndim == 3:
        img = img[..., 0]

    if img.ndim != 2:
        raise ValueError("Thermal image must be single-channel")

    h, w = img.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, float(angle), float(scale))

    out_img = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=interpolation,
        borderMode=border_mode
    )
    return out_img


# =====================================================
# GAUSSIAN NOISE
# =====================================================
def add_gaussian_noise(
    image: Union[str, np.ndarray],
    *,
    mean: float = 0.0,
    std: float = 25.0
) -> np.ndarray:
    
    """
    Add Gaussian noise to a thermal (single-channel) image.

    Parameters
    ----------
    image : str | np.ndarray
        Input thermal image or file path.

    mean : float
        Mean of the Gaussian noise.
        Units: pixel intensity
        Default: 0.0

    std : float
        Standard deviation of the Gaussian noise.
        Units: pixel intensity
        Default: 25.0

    Returns
    -------
    out_image : np.ndarray
        Noisy thermal image (same shape and dtype as input).
    """

    if std < 0:
        raise ValueError("std must be >= 0")

    if image is None:
        raise ValueError("Input image is None")

    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError("image must be path or numpy array")

    if img.size == 0:
        raise ValueError("Input image is empty")

    if img.ndim == 3:
        img = img[..., 0]

    if img.ndim != 2:
        raise ValueError("Thermal image must be single-channel")

    orig_dtype = img.dtype

    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise

    if orig_dtype == np.uint8:
        return np.clip(noisy, 0, 255).astype(np.uint8)

    if orig_dtype == np.uint16:
        return np.clip(noisy, 0, 65535).astype(np.uint16)

    return noisy.astype(orig_dtype)


# =====================================================
# SALT & PEPPER NOISE
# =====================================================
def add_salt_pepper_noise(
    image: Union[str, np.ndarray],
    *,
    density: float
) -> np.ndarray:
    
    """
    Add salt-and-pepper noise to a thermal (single-channel) image.

    PARAMETERS
    ----------
    image : str | np.ndarray
        Input thermal image or file path.

    density : float
        Fraction of pixels to corrupt with noise.

        Min / Max:
            0.0 ≤ density ≤ 1.0

        Units:
            Ratio (unitless)

        Best-case:
            0.001 – 0.01 for augmentation

    RETURNS
    -------
    noisy_image : np.ndarray
        Thermal image with salt & pepper noise applied.
    """

    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in range [0,1]")

    if image is None:
        raise ValueError("Input image is None")

    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError("image must be path or numpy array")

    if img.size == 0:
        raise ValueError("Input image is empty")

    if img.ndim == 3:
        img = img[..., 0]

    if img.ndim != 2:
        raise ValueError("Thermal image must be single-channel")

    noisy = img.copy()
    h, w = img.shape
    count = int(h * w * density)

    if img.dtype == np.uint8:
        salt, pepper = 255, 0
    elif img.dtype == np.uint16:
        salt, pepper = 65535, 0
    else:
        salt, pepper = img.max(), img.min()

    for _ in range(count):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        noisy[y, x] = salt if random.random() < 0.5 else pepper

    return noisy
