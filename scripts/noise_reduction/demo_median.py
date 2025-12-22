import cv2
import numpy as np

from thermal.noise_reduction import (
    thermal_median_filter,
    median_filter_score,
)


def main():
    image_path = "data/images/SingleChannel.png"

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    ksize = 3
    filtered = thermal_median_filter(original, ksize)

    scores = median_filter_score(original, filtered)

    vis = filtered.copy()

    cv2.putText(
        vis,
        f"Median Score: {scores['final']*100:.2f}%",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    stacked = np.hstack([original, vis])
    cv2.imshow("Original | Median Filter", stacked)

    cv2.imwrite("outputs/noise_reduction/images/median_result.png", stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
