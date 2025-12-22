import numpy as np
from skimage.metrics import structural_similarity as ssim

__all__ = ["evaluate_gaussian_scores", "overall_quality_score"]


def evaluate_gaussian_scores(
    original: np.ndarray,
    processed: np.ndarray
) -> dict:

    if original is None or processed is None:
        raise ValueError("Images must not be None")

    if original.shape != processed.shape:
        raise ValueError("Input and output must have same shape")

    def _to_float255(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img.astype(np.float32)
        if img.dtype == np.uint16:
            return (img.astype(np.float32) / 65535.0) * 255.0
        if np.issubdtype(img.dtype, np.floating):
            return np.clip(img, 0.0, 1.0).astype(np.float32) * 255.0
        raise TypeError(f"Unsupported dtype {img.dtype}")

    orig_f = _to_float255(original)
    proc_f = _to_float255(processed)

    data_range = orig_f.max() - orig_f.min()
    if data_range <= 0:
        raise ValueError("Invalid data range")

    ssim_val = ssim(orig_f, proc_f, data_range=data_range)

    mse = np.mean((orig_f - proc_f) ** 2)
    psnr = float("inf") if mse < 1e-10 else 10.0 * np.log10((data_range ** 2) / mse)

    return {"ssim": float(ssim_val), "psnr": float(psnr)}


def overall_quality_score(
    ssim_score: float,
    psnr_score: float,
    *,
    psnr_min: float = 20.0,
    psnr_max: float = 45.0,
    alpha: float = 0.7,
) -> float:

    if not (0.0 <= ssim_score <= 1.0):
        raise ValueError("SSIM must be in [0,1]")

    psnr_norm = (psnr_score - psnr_min) / (psnr_max - psnr_min)
    psnr_norm = float(np.clip(psnr_norm, 0.0, 1.0))

    return alpha * ssim_score + (1.0 - alpha) * psnr_norm
