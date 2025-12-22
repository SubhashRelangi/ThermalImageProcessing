import os
import cv2
import numpy as np

from thermal.ml_ready.thermal_argumentation import (
    thermal_flip,
    thermal_rotate,
    add_gaussian_noise,
    add_salt_pepper_noise
)

from thermal.ml_ready.thermal_argumentation_scores import (
    augmentation_difference_score
)


# =====================================================
# DEMO: THERMAL AUGMENTATION (ALL OPERATIONS)
# =====================================================
def main() -> None:

    input_path = "data/images/SingleChannel.png"
    output_dir = "outputs/ml_ready/images/thermal_augmentation"

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # LOAD IMAGE
    # -------------------------------
    original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Failed to load image: {input_path}")

    print("\n===== THERMAL AUGMENTATION DEMO =====")

    # =================================================
    # 1. FLIP (HORIZONTAL)
    # =================================================
    flip_h = thermal_flip(original, flip_code=1)
    scores = augmentation_difference_score(original, flip_h)

    cv2.imwrite(
        f"{output_dir}/flip_horizontal.png",
        np.hstack([original, flip_h])
    )

    print("\n[Flip Horizontal]")
    for k, v in scores.items():
        print(f"{k:>22}: {v:.4f}")

    # =================================================
    # 2. FLIP (VERTICAL)
    # =================================================
    flip_v = thermal_flip(original, flip_code=0)
    scores = augmentation_difference_score(original, flip_v)

    cv2.imwrite(
        f"{output_dir}/flip_vertical.png",
        np.hstack([original, flip_v])
    )

    print("\n[Flip Vertical]")
    for k, v in scores.items():
        print(f"{k:>22}: {v:.4f}")

    # =================================================
    # 3. ROTATION
    # =================================================
    rotated = thermal_rotate(
        original,
        angle=10.0,
        scale=1.0
    )

    scores = augmentation_difference_score(original, rotated)

    cv2.imwrite(
        f"{output_dir}/rotation_10deg.png",
        np.hstack([original, rotated])
    )

    print("\n[Rotation 10°]")
    for k, v in scores.items():
        print(f"{k:>22}: {v:.4f}")

    # =================================================
    # 4. GAUSSIAN NOISE
    # =================================================
    gaussian = add_gaussian_noise(
        original,
        mean=0.0,
        std=20.0
    )

    scores = augmentation_difference_score(original, gaussian)

    cv2.imwrite(
        f"{output_dir}/gaussian_noise.png",
        np.hstack([original, gaussian])
    )

    print("\n[Gaussian Noise]")
    for k, v in scores.items():
        print(f"{k:>22}: {v:.4f}")

    # =================================================
    # 5. SALT & PEPPER NOISE
    # =================================================
    sp_noise = add_salt_pepper_noise(
        original,
        density=0.02
    )

    scores = augmentation_difference_score(original, sp_noise)

    cv2.imwrite(
        f"{output_dir}/salt_pepper_noise.png",
        np.hstack([original, sp_noise])
    )

    print("\n[Salt & Pepper Noise]")
    for k, v in scores.items():
        print(f"{k:>22}: {v:.4f}")

    print("\n===== OUTPUT SAVED TO =====")
    print(output_dir)


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[DEMO ERROR] {e}")
