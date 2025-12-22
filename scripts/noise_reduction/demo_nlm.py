import cv2

from thermal.noise_reduction import (
    thermal_nlm_denoise,
    compute_nlm_score,
)


def main():
    image_path = "data/images/SingleChannel.png"

    original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original is None:
        raise FileNotFoundError("Failed to load input image")

    denoised = thermal_nlm_denoise(
        original,
        h=2.7,
        template_window=7,
        search_window=11,
    )

    scores = compute_nlm_score(original, denoised)

    vis = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        vis,
        f"NLM Score: {scores['final']*100:.2f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("NLM Denoised", vis)
    cv2.imwrite("outputs/noise_reduction/images/nlm_result.png", vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
