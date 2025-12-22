import cv2
import numpy as np

from thermal.noise_reduction import (
    thermal_gaussian_filter,
    evaluate_gaussian_scores,
    overall_quality_score,
)


def main():
    # --------------------------------------------------
    # LOAD INPUT IMAGE
    # --------------------------------------------------
    image_path = "data/images/SingleChannel.png"

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # --------------------------------------------------
    # APPLY GAUSSIAN FILTER
    # --------------------------------------------------
    gaussian = thermal_gaussian_filter(
        original,
        ksize=(5, 5),
        sigma_x=1.5,
        output_dtype=np.uint8,
    )

    # --------------------------------------------------
    # EVALUATE SCORES
    # --------------------------------------------------
    scores = evaluate_gaussian_scores(original, gaussian)

    oqs = overall_quality_score(
        ssim_score=scores["ssim"],
        psnr_score=scores["psnr"],
    )

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    vis = gaussian.copy()

    overlay_text = (
        f"OQS: {oqs * 100:.2f}% | "
        f"SSIM: {scores['ssim']:.3f} | "
        f"PSNR: {scores['psnr']:.2f} dB"
    )

    cv2.putText(
        vis,
        overlay_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,),
        2,
        cv2.LINE_AA,
    )

    stacked = np.hstack([original, vis])

    cv2.imshow("Original | Gaussian Filtered", stacked)

    # --------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------
    output_path = "outputs/noise_reduction/images/gaussian_result.png"
    cv2.imwrite(output_path, stacked)

    print(f"[INFO] Output saved to: {output_path}")
    print(f"[INFO] OQS: {oqs * 100:.2f}%")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
