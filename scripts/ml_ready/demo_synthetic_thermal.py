import cv2
import numpy as np

from thermal.ml_ready.synthetic_thermal import synthetic_thermal
from thermal.ml_ready.synthetic_thermal_scores import thermal_similarity_score


# =====================================================
# DEMO: SYNTHETIC THERMAL GENERATION
# =====================================================
def main() -> None:

    input_path = "data/images/image.png"
    output_path = "outputs/ml_ready/synthetic_thermal_demo.png"

    original = cv2.imread(input_path)
    if original is None:
        raise FileNotFoundError("Input image not found")

    synthetic = synthetic_thermal(
        original,
        rotate_deg=None,
        noise_std=4.0,
        hot_pixel_count=150
    )

    scores = thermal_similarity_score(original, synthetic)

    print("\n[Synthetic Thermal Similarity Scores]")
    for k, v in scores.items():
        print(f"{k:>14}: {v}")

    vis = cv2.putText(
        synthetic.copy(),
        f"Similarity: {scores['final_score']:.2f}%",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        255,
        2,
        cv2.LINE_AA
    )

    stacked = np.hstack([
        cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
        vis
    ])

    cv2.imshow("Original | Synthetic Thermal", stacked)
    cv2.imwrite(output_path, stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
