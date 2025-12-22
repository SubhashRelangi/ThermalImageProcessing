import cv2 as cv
import numpy as np
from typing import Dict

Image = np.ndarray

def scharr_edge_score(
    input_image: Image,
    edge_map: Image,
) -> Dict[str, float]:
    """
    Compute a composite Scharr edge enhancement score
    by comparing input image and output edge map.

    Metrics used:
        - Gradient Energy Gain
        - Edge IoU
        - Edge-to-Noise Ratio
        - Background Noise Penalty

    Returns
    -------
    scores : dict
        {
            "gradient_gain": float,
            "edge_iou": float,
            "enr": float,
            "noise_penalty": float,
            "final_score": float
        }
    """

    try:
        if input_image is None or edge_map is None:
            raise ValueError("Input or edge_map is None.")
        if input_image.shape != edge_map.shape:
            raise ValueError("Input and output must have same shape.")

        # --- Gradients on input ---
        gx_in = cv.Scharr(input_image, cv.CV_32F, 1, 0)
        gy_in = cv.Scharr(input_image, cv.CV_32F, 0, 1)
        grad_in = cv.magnitude(gx_in, gy_in)

        # --- Gradients on output ---
        gx_out = cv.Scharr(edge_map, cv.CV_32F, 1, 0)
        gy_out = cv.Scharr(edge_map, cv.CV_32F, 0, 1)
        grad_out = cv.magnitude(gx_out, gy_out)

        # 1 Gradient Energy Gain
        grad_gain = (np.mean(grad_out) + 1e-6) / (np.mean(grad_in) + 1e-6)
        grad_gain_n = min(grad_gain / 2.0, 1.0)

        # 2 Edge IoU
        edge_in = grad_in > np.percentile(grad_in, 90)
        edge_out = edge_map > 0
        intersection = np.logical_and(edge_in, edge_out).sum()
        union = np.logical_or(edge_in, edge_out).sum()
        edge_iou = intersection / (union + 1e-6)

        # 3 Edge-to-Noise Ratio
        edge_vals = input_image[edge_out]
        bg_vals = input_image[~edge_out]

        enr = (np.mean(edge_vals) + 1e-6) / (np.std(bg_vals) + 1e-6)
        enr_n = min(enr / 5.0, 1.0)

        # 4 Noise penalty
        noise_penalty = np.std(bg_vals) / (np.std(input_image) + 1e-6)
        noise_penalty = min(max(noise_penalty, 1.0), 2.0)

        # --- Final composite score ---
        final_score = 100.0 * (
            (0.35 * grad_gain_n) +
            (0.35 * edge_iou) +
            (0.30 * enr_n)
        ) / noise_penalty

        return {
            "gradient_gain": grad_gain,
            "edge_iou": edge_iou,
            "enr": enr,
            "noise_penalty": noise_penalty,
            "final_score": final_score
        }

    except Exception as e:
        raise RuntimeError(f"Score calculation failed: {e}") from e

