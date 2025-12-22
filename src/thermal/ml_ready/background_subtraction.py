import cv2
import numpy as np
import time
from typing import Union


# =====================================================
# VIDEO BACKGROUND SUBTRACTION (TEMPORAL MEDIAN)
# =====================================================
def video_background_subtraction(
    video: Union[str, cv2.VideoCapture],
    *,
    num_samples: int = 25,
    threshold: int = 30,
    seed: int | None = None,
    show: bool = True,
    exit_key: int = 27
) -> None:
    """
    Perform background subtraction on a video using temporal median estimation
    and play the foreground result live.

    PIPELINE
    --------
    Video
        → Random frame sampling
        → Temporal median background estimation
        → Frame-wise background subtraction
        → Live playback (foreground mask)

    PARAMETERS
    ----------
    video : str | cv2.VideoCapture
        Input video source.

    num_samples : int
        Number of frames used for background estimation.
        Default: 25

    threshold : int
        Absolute difference threshold (0–255).
        Default: 30

    seed : int | None
        Random seed for reproducibility.

    show : bool
        Display video windows.

    exit_key : int
        Key to exit playback (default: ESC).

    RETURNS
    -------
    None
    """

    start_time = time.time()

    try:
        # -------------------------------
        # VALIDATION
        # -------------------------------
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        if not (0 <= threshold <= 255):
            raise ValueError("threshold must be in range 0–255")

        if seed is not None:
            np.random.seed(seed)

        # -------------------------------
        # OPEN VIDEO
        # -------------------------------
        if isinstance(video, cv2.VideoCapture):
            cap = video
        elif isinstance(video, str):
            cap = cv2.VideoCapture(video)
        else:
            raise TypeError("video must be a file path or cv2.VideoCapture")

        if not cap.isOpened():
            raise IOError("Cannot open video source")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError("Video contains no frames")

        # -------------------------------
        # BACKGROUND ESTIMATION
        # -------------------------------
        frame_ids = np.random.randint(
            0, frame_count, size=min(num_samples, frame_count)
        )

        samples = []

        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            if not ret:
                continue

            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            samples.append(frame.astype(np.float32))

        if len(samples) == 0:
            raise RuntimeError("No valid frames collected for background")

        background = np.median(samples, axis=0)
        background = np.clip(background, 0, 255).astype(np.uint8)

        print(
            f"{'Background estimation time:':<36}"
            f"{time.time() - start_time:.4f} s"
        )

        # -------------------------------
        # PLAY VIDEO WITH BACKGROUND REMOVED
        # -------------------------------
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            diff = cv2.absdiff(gray, background)
            _, fg = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

            if show:
                stacked = np.hstack([
                    background,
                    gray,
                    fg
                ])
                cv2.imshow("Background | Frame | Foreground", stacked)

                if cv2.waitKey(1) & 0xFF == exit_key:
                    break

        cap.release()
        if show:
            cv2.destroyAllWindows()

    except Exception as e:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        raise RuntimeError(
            f"video_background_subtraction failed: {e}"
        ) from e


# =====================================================
# SPATIAL BACKGROUND SUBTRACTION (SINGLE FRAME)
# =====================================================
def spatial_background_subtraction(
    image: Union[str, np.ndarray],
    *,
    ksize: int = 51,
    sigma: float = 0.0
) -> np.ndarray:
    """
    Perform background subtraction using spatial Gaussian smoothing.

    PARAMETERS
    ----------
    image : str | np.ndarray
        Input grayscale image.

    ksize : int
        Gaussian kernel size (odd).
        Default: 51

    sigma : float
        Gaussian sigma.
        Default: 0.0 (auto)

    RETURNS
    -------
    residual : np.ndarray
        Foreground residual image.
    """

    start_time = time.time()

    try:
        if image is None:
            raise ValueError("Input image is None")

        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError("Cannot read image")
        else:
            img = np.asarray(image)

        if img.ndim != 2:
            raise ValueError("Input must be single-channel grayscale")

        if ksize <= 0:
            raise ValueError("ksize must be > 0")
        if ksize % 2 == 0:
            ksize += 1

        background = cv2.GaussianBlur(
            img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma
        )

        residual = cv2.absdiff(img, background)

        print(
            f"{'Spatial background time:':<36}"
            f"{time.time() - start_time:.4f} s"
        )

        return residual

    except Exception as e:
        raise RuntimeError(
            f"spatial_background_subtraction failed: {e}"
        ) from e
