import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _edge_strength_ratio(inp: np.ndarray, out: np.ndarray) -> float:
    gx_in = cv2.Sobel(inp, cv2.CV_32F, 1, 0)
    gy_in = cv2.Sobel(inp, cv2.CV_32F, 0, 1)
    gx_out = cv2.Sobel(out, cv2.CV_32F, 1, 0)
    gy_out = cv2.Sobel(out, cv2.CV_32F, 0, 1)

    mag_in = np.sqrt(gx_in**2 + gy_in**2)
    mag_out = np.sqrt(gx_out**2 + gy_out**2)

    return float(np.mean(mag_out) / (np.mean(mag_in) + 1e-8))


def _noise_residual_energy(inp: np.ndarray, out: np.ndarray) -> float:
    residual = inp.astype(np.float32) - out.astype(np.float32)
    return float(np.mean(residual**2))


def wavelet_composite_score(
    inp: np.ndarray,
    out: np.ndarray
) -> dict:
    """
    Composite quality score for wavelet-denoised thermal images.
    """

    if inp is None or out is None:
        raise ValueError("Input or output image is None")

    inp_u8 = (
        inp if inp.dtype == np.uint8
        else cv2.normalize(inp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    )
    out_u8 = (
        out if out.dtype == np.uint8
        else cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    )

    ssim_val = ssim(inp_u8, out_u8, data_range=255)

    edge_ratio = _edge_strength_ratio(inp_u8, out_u8)
    edge_score = np.clip((edge_ratio - 0.7) / (1.1 - 0.7), 0.0, 1.0)

    noise_energy = _noise_residual_energy(inp_u8, out_u8)
    noise_score = np.exp(-noise_energy / 50.0)

    final = (
        0.40 * ssim_val +
        0.35 * edge_score +
        0.25 * noise_score
    )

    return {
        "final": float(np.clip(final, 0.0, 1.0)),
        "ssim": float(ssim_val),
        "edge_ratio": float(edge_ratio),
        "noise_energy": float(noise_energy),
    }
