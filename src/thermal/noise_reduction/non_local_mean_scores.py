import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _edge_retention(inp: np.ndarray, out: np.ndarray) -> float:
    gx_in = cv2.Sobel(inp, cv2.CV_32F, 1, 0)
    gy_in = cv2.Sobel(inp, cv2.CV_32F, 0, 1)
    gx_out = cv2.Sobel(out, cv2.CV_32F, 1, 0)
    gy_out = cv2.Sobel(out, cv2.CV_32F, 0, 1)

    mag_in = np.sqrt(gx_in**2 + gy_in**2)
    mag_out = np.sqrt(gx_out**2 + gy_out**2)

    return float(
        np.mean(mag_out) / (np.mean(mag_in) + 1e-6)
    )


def _noise_reduction(inp: np.ndarray, out: np.ndarray) -> float:
    residual = inp.astype(np.float32) - out.astype(np.float32)
    var_in = np.var(inp.astype(np.float32))
    var_res = np.var(residual)
    return 0.0 if var_in == 0 else float(1.0 - var_res / var_in)


def _variance_reduction(inp: np.ndarray, out: np.ndarray) -> float:
    var_in = np.var(inp.astype(np.float32))
    var_out = np.var(out.astype(np.float32))
    return 0.0 if var_in == 0 else float((var_in - var_out) / var_in)


def compute_nlm_score(inp: np.ndarray, out: np.ndarray) -> dict:
    """
    Composite NLM quality score for thermal images.
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
    edge = _edge_retention(inp_u8, out_u8)
    noise = _noise_reduction(inp_u8, out_u8)
    var = _variance_reduction(inp_u8, out_u8)

    edge_norm = np.clip((edge - 0.7) / (1.1 - 0.7), 0.0, 1.0)

    final = (
        0.35 * ssim_val +
        0.25 * edge_norm +
        0.25 * noise +
        0.15 * var
    )

    return {
        "final": float(np.clip(final, 0.0, 1.0)),
        "ssim": float(ssim_val),
        "edge_ret": float(edge),
        "noise_red": float(noise),
        "var_red": float(var),
    }
