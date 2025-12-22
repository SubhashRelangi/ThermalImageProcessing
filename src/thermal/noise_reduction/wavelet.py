import time
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet

def wavelet_denoise_thermal(
    image,
    *,
    method: str = "BayesShrink",
    mode: str = "soft",
    wavelet: str = "db4",
    wavelet_levels: int = 2,
    rescale_sigma: bool = True,
    preserve_dtype: bool = True
) -> np.ndarray:

    start = time.perf_counter()

    try:
        # -------- Load --------
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("Input must be numpy array or file path")

        # -------- Validate --------
        if img.ndim == 2:
            pass
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img[..., 0]
        else:
            raise ValueError("Expected SINGLE-CHANNEL thermal image")

        orig_dtype = img.dtype

        # -------- Normalize --------
        if orig_dtype == np.uint8:
            img_f = img.astype(np.float32) / 255.0
        elif orig_dtype == np.uint16:
            img_f = img.astype(np.float32) / 65535.0
        elif np.issubdtype(orig_dtype, np.floating):
            img_f = np.clip(img.astype(np.float32), 0.0, 1.0)
        else:
            raise TypeError(f"Unsupported dtype: {orig_dtype}")

        # -------- Wavelet level safety --------
        h, w = img_f.shape
        max_levels = int(np.floor(np.log2(min(h, w))))
        wavelet_levels = max(1, min(wavelet_levels, max_levels))

        # -------- Denoise --------
        den_f = denoise_wavelet(
            img_f,
            method=method,
            mode=mode,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
            rescale_sigma=rescale_sigma,
            channel_axis=None
        )

        # -------- Restore dtype --------
        if preserve_dtype:
            if orig_dtype == np.uint16:
                out = (den_f * 65535).clip(0, 65535).astype(np.uint16)
            else:
                out = (den_f * 255).clip(0, 255).astype(np.uint8)
        else:
            out = den_f.astype(np.float32)

    except Exception as e:
        raise RuntimeError(f"[Wavelet Denoising Failed] {e}") from e

    finally:
        elapsed = time.perf_counter() - start
        print(f"[Wavelet Denoise] Time: {elapsed:.4f}s")

    return out

