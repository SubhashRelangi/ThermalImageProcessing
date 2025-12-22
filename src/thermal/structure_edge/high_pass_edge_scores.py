import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim


# =====================================================
# INTERNAL METRICS (ASSUME VALID INPUTS)
# =====================================================
def _mean_gradient(img: np.ndarray) -> float:
    gx = cv.Scharr(img, cv.CV_32F, 1, 0)
    gy = cv.Scharr(img, cv.CV_32F, 0, 1)
    return float(np.mean(cv.magnitude(gx, gy)))


def _edge_map(img: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    gx = cv.Scharr(img, cv.CV_32F, 1, 0)
    gy = cv.Scharr(img, cv.CV_32F, 0, 1)
    mag = cv.magnitude(gx, gy)
    return mag >= np.percentile(mag, percentile)


def _edge_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union > 0 else 0.0


def _noise_amplification(inp: np.ndarray, out: np.ndarray) -> float:
    diff = out.astype(np.float32) - inp.astype(np.float32)
    return float(np.std(diff) / (np.std(inp) + 1e-6))


# =====================================================
# COMPOSITE HIGH-PASS EDGE SCORE (SELF-VALIDATING)
# =====================================================
def highpass_edge_score(input_img: np.ndarray, output_img: np.ndarray) -> float:

    for name, img in {"Input": input_img, "Output": output_img}.items():
        if img is None:
            raise ValueError(f"{name} image is None")
        if not isinstance(img, np.ndarray):
            raise TypeError(f"{name} image must be numpy ndarray")
        if img.ndim != 2:
            raise ValueError(f"{name} image must be single-channel")
        if img.size == 0:
            raise ValueError(f"{name} image is empty")

    try:
        S = ssim(input_img, output_img, data_range=255)
        G = _mean_gradient(output_img) / (_mean_gradient(input_img) + 1e-6)
        E = _edge_iou(_edge_map(input_img), _edge_map(output_img))
        N = _noise_amplification(input_img, output_img)

        S_n = np.clip(S, 0.0, 1.0)
        G_n = np.clip((G - 1.0) / 1.5, 0.0, 1.0)
        E_n = np.clip(E, 0.0, 1.0)
        N_p = np.clip(N - 1.0, 0.0, 1.0)

        score = 100.0 * (
            0.35 * S_n +
            0.35 * G_n +
            0.20 * E_n -
            0.10 * N_p
        )

        return float(np.clip(score, 0.0, 100.0))

    except Exception as e:
        raise RuntimeError(f"highpass_edge_score failed: {e}")