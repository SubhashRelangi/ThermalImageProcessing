import cv2

from thermal.noise_reduction.temporal_average import total_temporal_average, recursive_temporal_average

def main():
    video_path = "data/Videos/video.mp4"

    total_avg = total_temporal_average(
        video_path,
        is_thermal=False,
        preserve_radiometric=False,
        max_frames=None,
        roi=None,
        output_dtype="uint8",
        units="raw",
        apply_normalization=True,
        norm_alpha=0,
        norm_beta=255,
        norm_type=cv2.NORM_MINMAX,
        show=False,
        verbose=True,
        alpha=0.02,               # present but ignored by total
        running_average_mode=True,  # incremental mean
    )
    rec_avg = recursive_temporal_average(
        video_path,
        is_thermal=False,
        preserve_radiometric=False,
        max_frames=None,
        roi=None,
        output_dtype="uint8",
        units="raw",
        apply_normalization=True,
        norm_alpha=0,
        norm_beta=255,
        norm_type=cv2.NORM_MINMAX,
        show=False,
        verbose=True,
        alpha=0.02,               # used by recursive
        running_average_mode=True,  # ignored by recursive
    )

    cv2.imshow("Total Temporal Average", total_avg)
    cv2.imshow("Recursive Temporal Average", rec_avg)

    cv2.imwrite("outputs/noise_reduction/images/total_temporal.png", total_avg)
    cv2.imwrite("outputs/noise_reduction/images/recursive_temporal.png", rec_avg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()