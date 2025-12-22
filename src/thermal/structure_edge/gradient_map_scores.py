import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_gradient_score(
    input_gray: np.ndarray,
    gradient_gray: np.ndarray,
) -> float:
    """
    Composite thermal gradient quality score (0–100).
    """

    # ---------- validation ----------
    for name, img in {"Input": input_gray, "Gradient": gradient_gray}.items():
        if img is None:
            raise ValueError(f"{name} image is None")
        if not isinstance(img, np.ndarray):
            raise TypeError(f"{name} image must be numpy ndarray")
        if img.ndim != 2:
            raise ValueError(f"{name} image must be single-channel")
        if img.size == 0:
            raise ValueError(f"{name} image is empty")

    try:
        # --- ensure uint8 ---
        inp_u8 = (
            input_gray if input_gray.dtype == np.uint8
            else np.clip(input_gray, 0, 255).astype(np.uint8)
        )
        grad_u8 = (
            gradient_gray if gradient_gray.dtype == np.uint8
            else np.clip(gradient_gray, 0, 255).astype(np.uint8)
        )

        # --- SSIM ---
        ssim_val = ssim(inp_u8, grad_u8, data_range=255)

        # --- Gradient energy ---
        gx_in = cv.Sobel(inp_u8, cv.CV_32F, 1, 0)
        gy_in = cv.Sobel(inp_u8, cv.CV_32F, 0, 1)
        mag_in = cv.magnitude(gx_in, gy_in)

        gx_out = cv.Sobel(grad_u8, cv.CV_32F, 1, 0)
        gy_out = cv.Sobel(grad_u8, cv.CV_32F, 0, 1)
        mag_out = cv.magnitude(gx_out, gy_out)

        energy_ratio = float(np.mean(mag_out) / (np.mean(mag_in) + 1e-6))
        energy_score = np.clip((energy_ratio - 0.7) / 0.6, 0.0, 1.0)

        # --- Flat-region noise penalty ---
        flat_mask = mag_in < np.percentile(mag_in, 20)
        noise_penalty = np.clip(np.std(mag_out[flat_mask]) / 255.0, 0.0, 1.0)

        final_score = (
            0.40 * np.clip(ssim_val, 0.0, 1.0) +
            0.30 * energy_score +
            0.30 * (1.0 - noise_penalty)
        )

        return float(np.clip(final_score, 0.0, 1.0) * 100.0)

    except Exception as e:
        raise RuntimeError(f"calculate_gradient_score failed: {e}")

