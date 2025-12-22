import cv2 as cv
import numpy as np

__all__ = [
    "calculate_low_pass_scores",
    "calculate_sharpen_scores",
    "calculate_laplacian_scores",
    "calculate_sobel_scores",
    "avg_score_subtract_low_pass",
    "avg_score_convolve_with_kernel",
    "avg_score_laplacian",
    "avg_score_sobel",
]


def high_frequency_energy(img: np.ndarray) -> float:
    lap = cv.Laplacian(img.astype(np.float32), cv.CV_32F)
    return float(np.mean(lap ** 2))


def edge_strength(img: np.ndarray) -> float:
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def edge_density(edge_map: np.ndarray) -> float:
    return float(np.count_nonzero(edge_map) / edge_map.size)


def calculate_low_pass_scores(inp: np.ndarray, out: np.ndarray) -> dict:
    return {
        "hf_energy_gain": high_frequency_energy(out) /
                          (high_frequency_energy(inp) + 1e-8),
        "edge_strength_gain": edge_strength(out) /
                              (edge_strength(inp) + 1e-8),
    }


def calculate_sharpen_scores(inp: np.ndarray, out: np.ndarray) -> dict:
    lap_in = cv.Laplacian(inp.astype(np.float32), cv.CV_32F).var()
    lap_out = cv.Laplacian(out.astype(np.float32), cv.CV_32F).var()
    return {"sharpness_gain": lap_out / (lap_in + 1e-8)}


def calculate_laplacian_scores(edge_map: np.ndarray) -> dict:
    return {
        "edge_density": edge_density(edge_map),
        "edge_strength": edge_strength(edge_map),
    }


def calculate_sobel_scores(grad: np.ndarray) -> dict:
    return {
        "mean_gradient": float(np.mean(grad)),
        "gradient_energy": float(np.mean(grad.astype(np.float32) ** 2)),
    }


def avg_score_subtract_low_pass(scores: dict) -> float:
    return float(
        (scores["hf_energy_gain"] / 1.5 +
         scores["edge_strength_gain"] / 1.4) / 2.0
    )


def avg_score_convolve_with_kernel(scores: dict) -> float:
    return float(scores["sharpness_gain"] / 8.0)


def avg_score_laplacian(scores: dict) -> float:
    return float(
        (scores["edge_density"] +
         scores["edge_strength"] / 80.0) / 2.0
    )


def avg_score_sobel(scores: dict) -> float:
    return float(
        (scores["mean_gradient"] / 20.0 +
         scores["gradient_energy"] / 600.0) / 2.0
    )
