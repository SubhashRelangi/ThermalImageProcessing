import cv2 as cv
import numpy as np

from thermal.structure_edge.upsharp_masking import (
    apply_unsharp_masking
)
from thermal.structure_edge.upsharp_masking_scores import (
    calculate_usm_score
)


# =====================================================
# DEMO: UNSHARP MASKING
# =====================================================
def main() -> None:
    """
    Demonstration script for Unsharp Masking (USM).

    - Loads a single-channel image
    - Applies unsharp masking
    - Computes USM quality score
    - Visualizes original vs enhanced result
    """

    input_path = "data/images/SingleChannel.png"
    output_path = "outputs/structure_edge/images/unsharp_masking_output.png"

    # -------------------------------
    # LOAD IMAGE
    # -------------------------------
    original = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Failed to load image: {input_path}")

    # -------------------------------
    # APPLY UNSHARP MASKING
    # -------------------------------
    enhanced = apply_unsharp_masking(
        original,
        blur_ksize=(5, 5),
        mask_scale=0.5,
        sharp_alpha=1.2
    )

    # -------------------------------
    # COMPUTE SCORE
    # -------------------------------
    score, details = calculate_usm_score(original, enhanced)
    print(f"[Unsharp Masking Score] {score * 100:.2f} / 100")

    # -------------------------------
    # DRAW SCORE
    # -------------------------------
    vis = enhanced.copy()

    cv.rectangle(vis, (5, 5), (460, 95), (0, 0, 0), -1)

    cv.putText(
        vis,
        f"USM Score: {score:.2f}",
        (10, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv.putText(
        vis,
        f"G:{details['gradient_gain']:.2f}  "
        f"L:{details['laplacian_ratio']:.2f}  "
        f"S:{details['ssim']:.2f}  "
        f"N:{details['noise_gain']:.2f}",
        (10, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1
    )

    # -------------------------------
    # STACK FOR COMPARISON
    # -------------------------------
    stacked = np.hstack([
        original,
        vis
    ])

    # -------------------------------
    # DISPLAY + SAVE
    # -------------------------------
    cv.imshow("Original | Unsharp Masking + Score", stacked)
    cv.imwrite(output_path, stacked)

    cv.waitKey(0)
    cv.destroyAllWindows()


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[DEMO ERROR] {e}")
