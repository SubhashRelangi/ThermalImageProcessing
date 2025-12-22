import cv2 as cv
import numpy as np

from thermal.structure_edge.gradient_map_scores import calculate_gradient_score
from thermal.structure_edge.gradient_map import gradient_map


# =====================================================
# DEMO: GRADIENT MAP
# =====================================================
def main() -> None:

    input_path = "data/images/SingleChannel.png"
    output_path = "outputs/structure_edge/images/gradient_map_output.png"

    # -------------------------------
    # LOAD IMAGE
    # -------------------------------
    image = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {input_path}")

    # -------------------------------
    # APPLY GRADIENT MAP
    # -------------------------------
    grad_vis = gradient_map(image)
    grad_gray = cv.cvtColor(grad_vis, cv.COLOR_BGR2GRAY)

    # -------------------------------
    # COMPUTE SCORE (FLOAT)
    # -------------------------------
    final_score = calculate_gradient_score(image, grad_gray)

    print(f"[Gradient Map Score] {final_score:.2f} / 100")

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    vis = grad_vis.copy()

    cv.putText(
        vis,
        f"Gradient Map Score: {final_score:.2f}/100",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv.LINE_AA
    )

    stacked = np.hstack((
        cv.cvtColor(image, cv.COLOR_GRAY2BGR),
        vis
    ))

    cv.imshow("Original | Gradient Map", stacked)
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
