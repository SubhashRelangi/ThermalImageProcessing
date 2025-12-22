import cv2 as cv
import numpy as np

from thermal.structure_edge.scarr import apply_scharr_operator
from thermal.structure_edge.scarr_scores import scharr_edge_score

# =====================================================
# DEMO: SCHARR EDGE DETECTION
# =====================================================
def main() -> None:
    """
    Demonstration script for Scharr edge detection and scoring.

    Loads a grayscale image, applies the Scharr operator,
    computes a composite edge quality score, and visualizes
    the results.
    """

    input_path = (
        "data/images/SingleChannel.png"
    )

    output_path = (
        "outputs/structure_edge/images/scarr_output.png"
    )

    # -------------------------------
    # LOAD IMAGE
    # -------------------------------
    image = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {input_path}")

    # -------------------------------
    # APPLY SCHARR OPERATOR
    # -------------------------------
    edge_map = apply_scharr_operator(
        image=image,
        blur_ksize=3,
        threshold_val=45.0
    )

    # -------------------------------
    # COMPUTE SCORE
    # -------------------------------
    scores = scharr_edge_score(image, edge_map)
    final_score = scores["final_score"]

    print(f"[Scharr Edge Score] {final_score:.2f} / 100")

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    vis = edge_map.copy()

    cv.putText(
        vis,
        f"Scharr Score: {final_score:.2f}/100",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv.LINE_AA
    )

    stacked = np.hstack((image, vis))
    cv.imshow("Original | Scharr Edge", stacked)
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
