import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy

__all__ = ["median_filter_score"]


def _edge_preservation(inp: np.ndarray, out: np.ndarray) -> float:
    lap_in = cv2.Laplacian(inp.astype(np.float32), cv2.CV_32F)
    lap_out = cv2.Laplacian(out.astype(np.float32), cv2.CV_32F)
    return float(
        np.clip(
            np.mean(np.abs(lap_out)) / (np.mean(np.abs(lap_in)) + 1e-6),
            0.0,
            1.0,
        )
    )


def _noise_removal(inp: np.ndarray, out: np.ndarray, k: float = 100.0) -> float:
    residual = inp.astype(np.float32) - out.astype(np.float32)
    var = np.var(residual)
    return float(np.clip(1.0 - np.exp(-var / k), 0.0, 1.0))


def _entropy_preservation(inp: np.ndarray, out: np.ndarray) -> float:
    return float(
        np.clip(
            shannon_entropy(out) / (shannon_entropy(inp) + 1e-6),
            0.0,
            1.0,
        )
    )


def median_filter_score(inp: np.ndarray, out: np.ndarray) -> dict:
    """
    Composite quality score for median filtering on thermal images.
    """

    if inp is None or out is None:
        raise ValueError("Input or output image is None")

    ssim_v = ssim(inp, out, data_range=255)
    edge_v = _edge_preservation(inp, out)
    noise_v = _noise_removal(inp, out)
    entropy_v = _entropy_preservation(inp, out)

    final = (
        0.40 * ssim_v +
        0.25 * edge_v +
        0.20 * noise_v +
        0.15 * entropy_v
    )

    return {
        "final": float(np.clip(final, 0.0, 1.0)),
        "ssim": float(ssim_v),
        "edge": edge_v,
        "noise": noise_v,
        "entropy": entropy_v,
    }
