import cv2

from thermal.ml_ready.background_subtraction import (
    spatial_background_subtraction,
    video_background_subtraction
)


def main() -> None:
    """
    Demo for background subtraction:
      - Spatial (single image)
      - Temporal (video playback with foreground)
    """

    # -------------------------------
    # SPATIAL BACKGROUND DEMO
    # -------------------------------
    image_path = "data/images/image.png"

    residual = spatial_background_subtraction(
        image_path,
        ksize=51
    )

    cv2.imshow("Spatial Residual", residual)
    cv2.imwrite(
        "outputs/ml_ready/images/background_subtraction/spatial_residual.png",
        residual
    )

    # -------------------------------
    # TEMPORAL VIDEO BACKGROUND DEMO
    # -------------------------------
    video_path = "data/Videos/resized_video_640x360.avi"

    video_background_subtraction(
        video_path,
        num_samples=30,
        threshold=30,
        seed=42,
        show=True
    )

    cv2.waitKey(40)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[DEMO ERROR] {e}")
