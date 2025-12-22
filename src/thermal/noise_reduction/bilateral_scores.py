import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def bilateral_effectiveness_score(
    inp: np.ndarray,
    out: np.ndarray
) -> float:

    if inp is None or out is None:
        raise ValueError("Input or output image is None")

    if inp.shape != out.shape:
        raise ValueError("Input and output images must have same shape")

    inp_f = inp.astype(np.float32)
    out_f = out.astype(np.float32)

    ssim_val = ssim(inp_f, out_f, data_range=255)

    def grad_mag(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        return np.sqrt(gx**2 + gy**2)

    edge = np.mean(grad_mag(out_f)) / (np.mean(grad_mag(inp_f)) + 1e-6)
    edge = np.clip(edge, 0, 1)

    lap_in = cv2.Laplacian(inp_f, cv2.CV_32F)
    lap_out = cv2.Laplacian(out_f, cv2.CV_32F)
    noise = 1.0 - (np.var(lap_out) / (np.var(lap_in) + 1e-6))
    noise = np.clip(noise, 0, 1)

    drift = abs(np.mean(out_f) - np.mean(inp_f)) / 255.0
    drift = np.clip(drift, 0, 1)

    return round(
        100.0 * np.clip(
            0.45 * ssim_val +
            0.25 * edge +
            0.20 * noise -
            0.10 * drift,
            0, 1
        ),
        2
    )
