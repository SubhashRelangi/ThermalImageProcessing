from typing import Optional, Tuple
import cv2 as cv
import time
import numpy as np

Image = np.ndarray


# =====================================================
# SUBTRACT LOW-PASS (SELF-CONTAINED)
# =====================================================
def subtract_low_pass(
    image: Image,
    *,
    gblur_ksize: Tuple[int, int] = (5, 5),
    sigma: float = 1.2,
    offset: float = 127.0,
    min_clip: int = 0,
    max_clip: int = 255
) -> Optional[Image]:
    
    """
    Enhance high-frequency content by subtracting a Gaussian low-pass image.

    ---------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------

    image : np.ndarray
        Description:
            Input grayscale image.

        Min & Max values:
            uint8 : [0, 255]

        Units:
            Pixel intensity

        Default values:
            None (required)

        Best case values:
            Image with smooth background and visible edges

    gblur_ksize : (int, int)
        Description:
            Gaussian blur kernel size.

        Min & Max values:
            Odd integers ≥ 3

        Units:
            Pixels

        Default values:
            (5, 5)

        Best case values:
            (5, 5)

    sigma : float
        Description:
            Gaussian standard deviation.

        Min & Max values:
            > 0

        Units:
            Pixels

        Default values:
            1.2

        Best case values:
            0.8 – 1.5

    offset : float
        Description:
            Bias added to avoid negative values.

        Min & Max values:
            [0, 255]

        Units:
            Intensity

        Default values:
            127.0

        Best case values:
            120 – 130

    min_clip / max_clip : int
        Description:
            Output intensity clipping range.

        Min & Max values:
            [0, 255]

        Units:
            Intensity

        Default values:
            0 / 255

        Best case values:
            0 / 255
    """


    start_time = time.time()

    try:
        # ---------- validation ----------
        if image is None:
            raise ValueError("Input image is None")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be numpy ndarray")

        if image.ndim == 3:
            image = image[..., 0]
        if image.ndim != 2 or image.size == 0:
            raise ValueError("Input must be non-empty single-channel image")

        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        # ensure odd kernel
        kx, ky = int(gblur_ksize[0]), int(gblur_ksize[1])
        if kx <= 0 or ky <= 0:
            raise ValueError("Kernel sizes must be positive")
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        gblur_ksize = (kx, ky)

        # ---------- processing ----------
        img_f = image.astype(np.float32)
        blurred = cv.GaussianBlur(img_f, gblur_ksize, sigma)

        high_pass = img_f - blurred + offset
        out = np.clip(high_pass, min_clip, max_clip).astype(np.uint8)

        print(f"[subtract_low_pass] Time: {time.time() - start_time:.6f}s")
        return out

    except Exception as e:
        raise RuntimeError(f"[subtract_low_pass] {e}") from e


# =====================================================
# CONVOLVE WITH KERNEL (SELF-CONTAINED)
# =====================================================
def convolve_with_kernel(
    image: Image,
    *,
    kernel: np.ndarray = np.array(
        [[0, -0.5, 0],
         [-0.5, 3.0, -0.5],
         [0, -0.5, 0]],
        dtype=np.float32
    ),
    ddepth: int = -1
) -> Optional[Image]:
    
    """
    Apply fixed spatial convolution for sharpening.

    ---------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------

    image : np.ndarray
        Description:
            Input grayscale image.

        Min & Max values:
            uint8 : [0, 255]

        Units:
            Pixel intensity

        Default values:
            None

        Best case values:
            Moderate contrast image

    kernel : np.ndarray
        Description:
            Convolution kernel.

        Min & Max values:
            Arbitrary real values

        Units:
            Weight coefficients

        Default values:
            Fixed 3×3 sharpening kernel

        Best case values:
            High-pass emphasis kernels

    ddepth : int
        Description:
            Output depth of filtered image.

        Min & Max values:
            OpenCV-supported depths

        Units:
            OpenCV enum

        Default values:
            -1 (same as input)

        Best case values:
            -1
    """

    start_time = time.time()

    try:
        # ---------- validation ----------
        if image is None:
            raise ValueError("Input image is None")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be numpy ndarray")

        if image.ndim == 3:
            image = image[..., 0]
        if image.ndim != 2 or image.size == 0:
            raise ValueError("Input must be single-channel image")

        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel must be square 2D matrix")

        # ---------- processing ----------
        img_f = image.astype(np.float32)
        filtered = cv.filter2D(img_f, ddepth=ddepth, kernel=kernel)

        out = np.clip(filtered, 0, 255).astype(np.uint8)

        print(f"[convolve_with_kernel] Time: {time.time() - start_time:.6f}s")
        return out

    except Exception as e:
        raise RuntimeError(f"[convolve_with_kernel] {e}") from e


# =====================================================
# LAPLACIAN EDGE DETECTOR (SELF-CONTAINED)
# =====================================================
def apply_laplacian_detector(
    image: Image,
    *,
    ksize: int = 3,
    ddepth: int = cv.CV_32F
) -> Optional[Image]:
    
    """
    Detect edges using Laplacian operator.

    ---------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------

    image : np.ndarray
        Description:
            Input grayscale image.

        Min & Max values:
            uint8 : [0, 255]

        Units:
            Pixel intensity

        Default values:
            None

        Best case values:
            Pre-smoothed image

    ksize : int
        Description:
            Laplacian kernel size.

        Min & Max values:
            Odd integers ≥ 1

        Units:
            Pixels

        Default values:
            3

        Best case values:
            3

    ddepth : int
        Description:
            Output depth.

        Min & Max values:
            OpenCV depth enums

        Units:
            OpenCV enum

        Default values:
            cv.CV_64F

        Best case values:
            cv.CV_64F
    """

    start_time = time.time()

    try:
        # ---------- validation ----------
        if image is None:
            raise ValueError("Input image is None")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be numpy ndarray")

        if image.ndim == 3:
            image = image[..., 0]
        if image.ndim != 2 or image.size == 0:
            raise ValueError("Input must be single-channel image")

        if ksize < 1 or ksize % 2 == 0:
            raise ValueError("ksize must be odd >= 1")

        # ---------- processing ----------
        img_f = image.astype(np.float32)
        img_blur = cv.GaussianBlur(img_f, (3, 3), 0.8)

        lap = cv.Laplacian(img_blur, ddepth=ddepth, ksize=ksize)
        lap_abs = np.abs(lap)

        out = cv.normalize(lap_abs, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        print(f"[apply_laplacian_detector] Time: {time.time() - start_time:.6f}s")
        return out

    except Exception as e:
        raise RuntimeError(f"[apply_laplacian_detector] {e}") from e


# =====================================================
# SOBEL X + Y EDGE DETECTOR (SELF-CONTAINED)
# =====================================================
def apply_sobel_xy_detectors(
    image: Image,
    *,
    ksize: int = 3,
    ddepth: int = cv.CV_32F,
    dx_sobel_x: int = 1,
    dy_sobel_x: int = 0,
    dx_sobel_y: int = 0,
    dy_sobel_y: int = 1,
) -> Optional[Image]:
    
    """
    Compute gradient magnitude using Sobel X and Y operators.

    ---------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------

    image : np.ndarray
        Description:
            Input grayscale image.

    ksize : int
        Description:
            Sobel kernel size.

        Min & Max values:
            Odd integers ≥ 3

        Units:
            Pixels

        Default values:
            3

        Best case values:
            3
    """

    start_time = time.time()

    try:
        # ---------- validation ----------
        if image is None:
            raise ValueError("Input image is None")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be numpy ndarray")

        if image.ndim == 3:
            image = image[..., 0]
        if image.ndim != 2 or image.size == 0:
            raise ValueError("Input must be single-channel image")

        if ksize < 3 or ksize % 2 == 0:
            raise ValueError("ksize must be odd >= 3")

        # ---------- processing ----------
        img_f = image.astype(np.float32)

        sobelx = cv.Sobel(img_f, ddepth, dx_sobel_x, dy_sobel_x, ksize=ksize)
        sobely = cv.Sobel(img_f, ddepth, dx_sobel_y, dy_sobel_y, ksize=ksize)

        magnitude = cv.magnitude(sobelx, sobely)
        out = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        print(f"[apply_sobel_xy_detectors] Time: {time.time() - start_time:.6f}s")
        return out

    except Exception as e:
        raise RuntimeError(f"[apply_sobel_xy_detectors] {e}") from e
