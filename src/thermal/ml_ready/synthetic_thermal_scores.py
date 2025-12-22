import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance


# =====================================================
# THERMAL SIMILARITY SCORE
# =====================================================
def thermal_similarity_score(
    original_bgr: np.ndarray,
    synthetic_thermal: np.ndarray
) -> dict:
    """
    Compute perceptual similarity between a visible-spectrum image
    (luminance reference) and a synthetic thermal image.

    Returns a composite similarity score along with individual metrics.
    """

    try:
        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------
        if original_bgr is None or synthetic_thermal is None:
            raise ValueError("Input images must not be None")

        if not isinstance(original_bgr, np.ndarray):
            raise TypeError("original_bgr must be numpy ndarray")

        if not isinstance(synthetic_thermal, np.ndarray):
            raise TypeError("synthetic_thermal must be numpy ndarray")

        if original_bgr.ndim != 3 or original_bgr.shape[2] != 3:
            raise ValueError("original_bgr must be a BGR image (H, W, 3)")

        if synthetic_thermal.ndim == 3:
            synthetic = cv2.cvtColor(synthetic_thermal, cv2.COLOR_BGR2GRAY)
        elif synthetic_thermal.ndim == 2:
            synthetic = synthetic_thermal
        else:
            raise ValueError("synthetic_thermal must be 2D or convertible to grayscale")

        # -------------------------------------------------
        # LUMINANCE REFERENCE
        # -------------------------------------------------
        ycrcb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
        reference = ycrcb[..., 0].astype(np.float32)
        synthetic = synthetic.astype(np.float32)

        if reference.shape != synthetic.shape:
            raise ValueError("Shape mismatch between reference and synthetic")

        # -------------------------------------------------
        # SSIM
        # -------------------------------------------------
        data_range = synthetic.max() - synthetic.min()
        if data_range <= 0:
            raise ValueError("Invalid data range for SSIM computation")

        ssim_val = ssim(reference, synthetic, data_range=data_range)
        ssim_n = np.clip(ssim_val, 0.0, 1.0)

        # -------------------------------------------------
        # GRADIENT SIMILARITY
        # -------------------------------------------------
        grad_r = cv2.magnitude(
            cv2.Sobel(reference, cv2.CV_32F, 1, 0, 3),
            cv2.Sobel(reference, cv2.CV_32F, 0, 1, 3),
        )
        grad_s = cv2.magnitude(
            cv2.Sobel(synthetic, cv2.CV_32F, 1, 0, 3),
            cv2.Sobel(synthetic, cv2.CV_32F, 0, 1, 3),
        )

        grad_sim = 1.0 - (
            np.mean(np.abs(grad_r - grad_s)) /
            (np.mean(grad_r) + 1e-6)
        )
        grad_n = np.clip(grad_sim, 0.0, 1.0)

        # -------------------------------------------------
        # LAPLACIAN VARIANCE RATIO
        # -------------------------------------------------
        lap_r = cv2.Laplacian(reference, cv2.CV_32F).var()
        lap_s = cv2.Laplacian(synthetic, cv2.CV_32F).var()
        lap_n = np.exp(-abs(1.0 - lap_s / (lap_r + 1e-6)))

        # -------------------------------------------------
        # HISTOGRAM DISTANCE
        # -------------------------------------------------
        hist_r, _ = np.histogram(reference, 256, (0, 255), density=True)
        hist_s, _ = np.histogram(synthetic, 256, (0, 255), density=True)
        hist_n = np.exp(-wasserstein_distance(hist_r, hist_s))

        # -------------------------------------------------
        # NOISE CONSISTENCY
        # -------------------------------------------------
        noise_r = reference - cv2.GaussianBlur(reference, (0, 0), 3)
        noise_s = synthetic - cv2.GaussianBlur(synthetic, (0, 0), 3)
        noise_n = np.exp(
            -abs(1.0 - np.var(noise_s) / (np.var(noise_r) + 1e-6))
        )

        # -------------------------------------------------
        # FINAL COMPOSITE SCORE
        # -------------------------------------------------
        final_score = (
            0.35 * ssim_n +
            0.25 * grad_n +
            0.15 * lap_n +
            0.15 * hist_n +
            0.10 * noise_n
        ) * 100.0

        return {
            "final_score": round(float(final_score), 2),
            "ssim": round(float(ssim_n), 3),
            "gradient": round(float(grad_n), 3),
            "laplacian": round(float(lap_n), 3),
            "histogram": round(float(hist_n), 3),
            "noise": round(float(noise_n), 3),
        }

    except Exception as e:
        raise RuntimeError(f"Thermal similarity scoring failed: {e}") from e
