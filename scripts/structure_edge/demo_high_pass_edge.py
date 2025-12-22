import cv2 as cv
import numpy as np

from thermal.structure_edge.high_pass_edge import HighPassEdge
from thermal.structure_edge.high_pass_edge_scores import highpass_edge_score


# =====================================================
# QUICK DRAW SCORE (LOCAL – DEMO ONLY)
# =====================================================
def draw_highpass_score(
    image: np.ndarray,
    score: float,
    label: str = "High-Pass Edge Score"
) -> np.ndarray:
    """
    Fast score overlay utility for demo use.
    """
    if image.ndim == 2:
        vis = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    text = f"{label}: {score:.1f} / 100"

    cv.putText(
        vis,
        text,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA
    )

    return vis


# =====================================================
# DEMO: HIGH-PASS EDGE ENHANCEMENT
# =====================================================
def main() -> None:

    input_path = "data/images/SingleChannel.png"
    output_path = "outputs/structure_edge/images/high_pass_edge_output.png"

    # -------------------------------
    # LOAD IMAGE
    # -------------------------------
    image = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {input_path}")

    # -------------------------------
    # APPLY HIGH-PASS EDGE ENHANCEMENT
    # -------------------------------
    enhanced = HighPassEdge(
        image,
        filter_mode="gaussian_hp",
        radius=40,
        gain=1.35,
        sigma=28.0,
        order=3,
        fft_flag=cv.DFT_COMPLEX_OUTPUT
    )

    # -------------------------------
    # COMPUTE SCORE
    # -------------------------------
    score = highpass_edge_score(image, enhanced)
    print(f"[High-Pass Edge Score] {score:.2f} / 100")

    # -------------------------------
    # VISUALIZE
    # -------------------------------
    enhanced_vis = draw_highpass_score(enhanced, score)

    stacked = np.hstack([
        cv.cvtColor(image, cv.COLOR_GRAY2BGR),
        enhanced_vis
    ])

    cv.imshow("Original | High-Pass Edge + Score", stacked)
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
