import cv2

from thermal.noise_reduction import (
    wavelet_denoise_thermal,
    wavelet_composite_score,
)


def main():
    image_path = "data/images/SingleChannel.png"

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError("Failed to load input image")

    denoised = wavelet_denoise_thermal(
        original,
        wavelet="db4",
        wavelet_levels=2,
    )

    scores = wavelet_composite_score(original, denoised)

    vis = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        vis,
        f"Wavelet Score: {scores['final']*100:.2f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Wavelet Denoised", vis)
    cv2.imwrite("outputs/noise_reduction/images/wavelet_result.png", vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
