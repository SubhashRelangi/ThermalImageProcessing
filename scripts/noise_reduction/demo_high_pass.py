import cv2
import numpy as np

from thermal.noise_reduction.high_pass import subtract_low_pass, convolve_with_kernel, apply_laplacian_detector, apply_sobel_xy_detectors
from thermal.noise_reduction.high_pass_scores import (
    calculate_low_pass_scores,
    calculate_sharpen_scores,
    calculate_laplacian_scores,
    calculate_sobel_scores,
    avg_score_subtract_low_pass,
    avg_score_convolve_with_kernel,
    avg_score_laplacian,
    avg_score_sobel,
)

def main():
    image = cv2.imread("data/images/SingleChannel.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not loaded")

    low = subtract_low_pass(image)
    fil = convolve_with_kernel(
        image,
        kernel=np.array(
            [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]],
            dtype=np.float32,
        ),
    )
    lap = apply_laplacian_detector(image)
    sob = apply_sobel_xy_detectors(image)

    low_f = avg_score_subtract_low_pass(calculate_low_pass_scores(image, low))
    fil_f = avg_score_convolve_with_kernel(calculate_sharpen_scores(image, fil))
    lap_f = avg_score_laplacian(calculate_laplacian_scores(lap))
    sob_f = avg_score_sobel(calculate_sobel_scores(sob))

    print("Low-pass :", low_f * 100)
    print("Kernel   :", fil_f * 100)
    print("Laplacian:", lap_f * 100)
    print("Sobel    :", sob_f * 100)

    cv2.imshow("Original", image)
    cv2.imshow("Low-pass", low)
    cv2.imshow("Kernel", fil)
    cv2.imshow("Laplacian", lap)
    cv2.imshow("Sobel", sob)
    cv2.imwrite("outputs/noise_reduction/images/output_low_pass.png", low)
    cv2.imwrite("outputs/noise_reduction/images/output_kernel.png", fil)
    cv2.imwrite("outputs/noise_reduction/images/output_laplacian.png", lap)
    cv2.imwrite("outputs/noise_reduction/images/output_sobel.png", sob)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
