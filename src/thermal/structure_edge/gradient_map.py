import cv2 as cv
import numpy as np
import time
from typing import Union

ImageArray = np.ndarray

def gradient_map(
    gray_image: Union[str, ImageArray],
    *,
    dx: int = 1,
    dy: int = 1,
    ddepth: int = cv.CV_32F,
    ksize: int = 3,
    sobel_scale: float = 0.8,
    sobel_delta: float = 0.0,
    sobel_border: int = cv.BORDER_DEFAULT,
    normalize_output: bool = True,
    colormap: int = cv.COLORMAP_JET,
) -> ImageArray:
    """
    Sobel gradient magnitude → normalize → colorize.
    """

    start_time = time.time()

    try:
        # ---------- load ----------
        if isinstance(gray_image, str):
            img = cv.imread(gray_image, cv.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(gray_image)
        else:
            img = gray_image

        # ---------- validation ----------
        if img is None:
            raise ValueError("Input image is None")
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image must be numpy ndarray")
        if img.ndim != 2:
            raise ValueError("Input image must be single-channel")
        if img.size == 0:
            raise ValueError("Input image is empty")

        if ksize < 1 or ksize % 2 == 0:
            raise ValueError("ksize must be odd >= 1")
        if sobel_scale <= 0:
            raise ValueError("sobel_scale must be > 0")

        img_f = img.astype(np.float32)

        # ---------- sobel ----------
        grad_x = cv.Sobel(
            img_f, ddepth, dx, 0,
            ksize=ksize,
            scale=sobel_scale,
            delta=sobel_delta,
            borderType=sobel_border
        )

        grad_y = cv.Sobel(
            img_f, ddepth, 0, dy,
            ksize=ksize,
            scale=sobel_scale,
            delta=sobel_delta,
            borderType=sobel_border
        )

        magnitude = cv.magnitude(grad_x, grad_y)

        # ---------- normalize ----------
        if normalize_output:
            magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        mag_u8 = np.clip(magnitude, 0, 255).astype(np.uint8)
        out = cv.applyColorMap(mag_u8, colormap)

        print(f"[gradient_map] Time: {time.time() - start_time:.6f}s")
        return out

    except Exception as e:
        raise RuntimeError(f"gradient_map failed: {e}")

