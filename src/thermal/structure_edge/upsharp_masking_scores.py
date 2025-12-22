import cv2 as cv
import numpy as np
from typing import Tuple, Dict
from skimage.metrics import structural_similarity as ssim

def calculate_usm_score(
    original: np.ndarray,
    processed: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite USM quality score.
    """

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------
    for name, img in {"Original": original, "Processed": processed}.items():
        if img is None:
            raise ValueError(f"{name} image is None")
        if not isinstance(img, np.ndarray):
            raise TypeError(f"{name} image must be numpy ndarray")
        if img.ndim == 3:
            img = img[..., 0]
        if img.ndim != 2 or img.size == 0:
            raise ValueError(f"{name} image must be single-channel")

    if original.shape != processed.shape:
        raise ValueError("Input and output shapes must match")

    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    # -------------------------------------------------
    # GRADIENT MAGNITUDE
    # -------------------------------------------------
    gx_o = cv.Sobel(original, cv.CV_32F, 1, 0, ksize=3)
    gy_o = cv.Sobel(original, cv.CV_32F, 0, 1, ksize=3)
    grad_o = cv.magnitude(gx_o, gy_o)

    gx_p = cv.Sobel(processed, cv.CV_32F, 1, 0, ksize=3)
    gy_p = cv.Sobel(processed, cv.CV_32F, 0, 1, ksize=3)
    grad_p = cv.magnitude(gx_p, gy_p)

    grad_gain = (np.mean(grad_p) + 1e-6) / (np.mean(grad_o) + 1e-6)

    # -------------------------------------------------
    # LAPLACIAN VARIANCE
    # -------------------------------------------------
    lap_o = cv.Laplacian(original, cv.CV_32F)
    lap_p = cv.Laplacian(processed, cv.CV_32F)
    lap_ratio = (np.var(lap_p) + 1e-6) / (np.var(lap_o) + 1e-6)

    # -------------------------------------------------
    # SSIM
    # -------------------------------------------------
    ssim_val = float(ssim(original, processed, data_range=255))

    # -------------------------------------------------
    # NOISE AMPLIFICATION (FLAT REGIONS)
    # -------------------------------------------------
    flat_mask = grad_o < np.percentile(grad_o, 25)
    noise_gain = (
        np.std(processed[flat_mask]) + 1e-6
    ) / (
        np.std(original[flat_mask]) + 1e-6
    )

    # -------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------
    G = np.clip((grad_gain - 1.0) / 0.35, 0.0, 1.0)
    L = np.clip((lap_ratio - 1.0) / 0.30, 0.0, 1.0)
    S = np.clip((ssim_val - 0.85) / 0.15, 0.0, 1.0)
    N = np.clip((noise_gain - 1.0) / 0.25, 0.0, 1.0)

    final_score = (
        0.35 * G +
        0.25 * L +
        0.30 * S -
        0.20 * N
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    details = {
        "gradient_gain": float(grad_gain),
        "laplacian_ratio": float(lap_ratio),
        "ssim": float(ssim_val),
        "noise_gain": float(noise_gain),
    }

    return final_score, details

