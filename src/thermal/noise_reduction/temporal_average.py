from typing import Tuple, Optional
import os
import cv2 as cv
import numpy as np
import time


# =====================================================
# TOTAL TEMPORAL AVERAGE
# =====================================================
def total_temporal_average(
    video_path: str,
    *,
    is_thermal: bool = False,
    preserve_radiometric: bool = False,
    max_frames: Optional[int] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    output_dtype: str = "uint8",
    units: str = "raw",
    apply_normalization: bool = True,
    norm_alpha: float = 0.0,
    norm_beta: float = 255.0,
    norm_type: int = cv.NORM_MINMAX,
    show: bool = False,
    verbose: bool = False,
    alpha: float = 0.02,                # unused (API parity)
    running_average_mode: bool = True
) -> np.ndarray:
    
    """
    Full-sequence temporal averaging (incremental mean).

    Args (types & meaning):

      video_path (str):
          Path to input video file.
          Min/Max: valid file path.
          Units: filesystem path.
          Default: required.
          Best case: Local SSD path.

      is_thermal (bool):
          Whether frames represent radiometric thermal data.
          Min/Max: True/False.
          Default: False.
          Best case: True for calibrated thermal processing.

      preserve_radiometric (bool):
          Keep raw float radiometric values (no normalization).
          Min/Max: True/False.
          Default: False.
          Best case: True when further scientific analysis is needed.

      max_frames (Optional[int]):
          Number of frames to process.
          Min: 1, Max: video length.
          Units: frames.
          Default: None (all frames).
          Best case: small value for testing, None for full accuracy.

      roi (Optional[Tuple[int,int,int,int]]):
          Region of interest as (x,y,w,h).
          Min/Max: must fit inside frame.
          Units: pixels.
          Default: None.
          Best case: cropped stable background.

      output_dtype (str):
          Output data type.
          Options: "uint8", "float32".
          Default: "uint8".
          Best case: "float32" for precision.

      units (str):
          Data unit label.
          Options: "raw", "celsius", "kelvin".
          Default: "raw".
          Best case: match camera specification.

      apply_normalization (bool):
          Normalize output using cv.normalize.
          Default: True.
          Best case: True for display, False for raw analysis.

      norm_alpha (float):
          Normalization lower bound.
          Default: 0.0.
          Units: intensity.

      norm_beta (float):
          Normalization upper bound.
          Default: 255.0.
          Units: intensity.

      norm_type (int):
          cv.normalize method.
          Default: cv.NORM_MINMAX.

      show (bool):
          Display debug windows.
          Default: False.

      verbose (bool):
          Add verbose info to metadata.
          Default: False.

      alpha (float):
          Unused in total averaging (kept for API uniformity).
          Default: 0.02.

      running_average_mode (bool):
          True → incremental mean, False → sum accumulation.
          Default: True.

    Returns:
      avg_frame (np.ndarray | None):
          Final averaged frame (uint8 or float32).
      metadata (dict):
          All processing statistics and error flags.
    """

    """
    Compute temporal average across video using an incremental running mean
    (or accumulation if running_average_mode=False).

    """

    # ---------- validation ----------
    if not isinstance(video_path, str):
        raise TypeError("video_path must be a string")

    if not os.path.exists(video_path):
        raise IOError(f"Video file not found: {video_path}")

    if output_dtype not in ("uint8", "float32"):
        raise ValueError("output_dtype must be 'uint8' or 'float32'")

    if units not in ("raw", "celsius", "kelvin"):
        raise ValueError("units must be 'raw', 'celsius', or 'kelvin'")

    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1 or None")

    if roi is not None:
        if len(roi) != 4 or any(v < 0 for v in roi):
            raise ValueError("roi must be (x, y, w, h) with non-negative values")

    start_time = time.perf_counter()
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Unable to open video")

    accumulator = None
    frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---- ROI ----
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w]

            if frame is None or not isinstance(frame, np.ndarray):
                raise RuntimeError("Invalid frame read from video")

            # ---- grayscale conversion (IN CORE) ----
            if frame.ndim == 2:
                cur = frame.astype(np.float32)
            elif frame.ndim == 3:
                cur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)
            else:
                raise ValueError(f"Unsupported frame dimensions: {frame.ndim}")

            # ---- accumulation ----
            if accumulator is None:
                accumulator = cur.astype(np.float64)
                frames = 1
            else:
                frames += 1
                if running_average_mode:
                    accumulator += (cur - accumulator) / frames
                else:
                    accumulator += cur

            if max_frames and frames >= max_frames:
                break

            if show:
                disp = cv.normalize(accumulator, None, 0, 255, cv.NORM_MINMAX)
                cv.imshow("Total Temporal Average", disp.astype(np.uint8))
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        if frames == 0 or accumulator is None:
            raise RuntimeError("No frames processed")

        avg_float = accumulator.astype(np.float32)

        if preserve_radiometric:
            return avg_float

        if apply_normalization:
            avg_float = cv.normalize(avg_float, None, norm_alpha, norm_beta, norm_type)

        if output_dtype == "uint8":
            return np.clip(avg_float, 0, 255).astype(np.uint8)

        return avg_float.astype(np.float32)

    finally:
        cap.release()
        if show:
            cv.destroyAllWindows()
        if verbose:
            print(f"[total_temporal_average] Time: {time.perf_counter() - start_time:.6f}s")


# =====================================================
# RECURSIVE TEMPORAL AVERAGE (EMA)
# =====================================================
def recursive_temporal_average(
    video_path: str,
    *,
    is_thermal: bool = False,
    preserve_radiometric: bool = False,
    max_frames: Optional[int] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    output_dtype: str = "uint8",
    units: str = "raw",
    apply_normalization: bool = True,
    norm_alpha: float = 0.0,
    norm_beta: float = 255.0,
    norm_type: int = cv.NORM_MINMAX,
    show: bool = False,
    verbose: bool = False,
    alpha: float = 0.02,
    running_average_mode: bool = True   # ignored
) -> np.ndarray:
    
    """
    Recursive exponential-moving-average over frames:
        O_new = (1 - alpha) * O_old + alpha * I_k

    Same parameter list as total_temporal_average (alpha used here).

    Args (types & meaning):

      video_path (str):
          Path to input video file.
          Min/Max: valid file path.
          Units: filesystem path.
          Default: required.
          Best case: Local SSD path.

      is_thermal (bool):
          Whether frames represent radiometric thermal data.
          Min/Max: True/False.
          Default: False.
          Best case: True for calibrated thermal processing.

      preserve_radiometric (bool):
          Keep raw float radiometric values (no normalization).
          Min/Max: True/False.
          Default: False.
          Best case: True when further scientific analysis is needed.

      max_frames (Optional[int]):
          Number of frames to process.
          Min: 1, Max: video length.
          Units: frames.
          Default: None (all frames).
          Best case: small value for testing, None for full accuracy.

      roi (Optional[Tuple[int,int,int,int]]):
          Region of interest as (x,y,w,h).
          Min/Max: must fit inside frame.
          Units: pixels.
          Default: None.
          Best case: cropped stable background.

      output_dtype (str):
          Output data type.
          Options: "uint8", "float32".
          Default: "uint8".
          Best case: "float32" for precision.

      units (str):
          Data unit label.
          Options: "raw", "celsius", "kelvin".
          Default: "raw".
          Best case: match camera specification.

      apply_normalization (bool):
          Normalize output using cv.normalize.
          Default: True.
          Best case: True for display, False for raw analysis.

      norm_alpha (float):
          Normalization lower bound.
          Default: 0.0.
          Units: intensity.

      norm_beta (float):
          Normalization upper bound.
          Default: 255.0.
          Units: intensity.

      norm_type (int):
          cv.normalize method.
          Default: cv.NORM_MINMAX.

      show (bool):
          Display debug windows.
          Default: False.

      verbose (bool):
          Add verbose info to metadata.
          Default: False.

      alpha (float):
          Exponential filter weight used in recursive update:
              O_new = (1 - alpha) * O_old + alpha * I_k
          Min/Max: (0.0, 1.0]
          Units: unitless
          Default: 0.02
          Best case: 0.01-0.05 for stable background; larger for faster adaptation.

      running_average_mode (bool):
          Present for API uniformity; ignored by recursive method.
          Default: True.

    Returns:
      avg_frame (np.ndarray | None):
          Final recursively averaged frame (uint8 or float32).
      metadata (dict):
          All processing statistics and error flags.
    """

    if not os.path.exists(video_path):
        raise IOError(f"Video file not found: {video_path}")

    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    start_time = time.perf_counter()
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Unable to open video")

    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Video contains no frames")

        if roi is not None:
            x, y, w, h = roi
            frame = frame[y:y + h, x:x + w]

        # ---- grayscale conversion (IN CORE) ----
        if frame.ndim == 2:
            accumulator = frame.astype(np.float64)
        elif frame.ndim == 3:
            accumulator = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float64)
        else:
            raise ValueError(f"Unsupported frame dimensions: {frame.ndim}")

        frames = 1
        w_old = 1.0 - alpha
        w_new = alpha

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if roi is not None:
                frame = frame[y:y + h, x:x + w]

            if frame.ndim == 2:
                cur = frame.astype(np.float32)
            elif frame.ndim == 3:
                cur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)
            else:
                raise ValueError(f"Unsupported frame dimensions: {frame.ndim}")

            if cur.shape != accumulator.shape:
                raise RuntimeError("Frame shape changed during processing")

            accumulator = cv.addWeighted(
                accumulator, w_old,
                cur.astype(np.float64), w_new,
                0.0
            )

            frames += 1
            if max_frames and frames >= max_frames:
                break

            if show:
                disp = cv.normalize(accumulator, None, 0, 255, cv.NORM_MINMAX)
                cv.imshow("Recursive Temporal Average", disp.astype(np.uint8))
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        avg_float = accumulator.astype(np.float32)

        if preserve_radiometric:
            return avg_float

        if apply_normalization:
            avg_float = cv.normalize(avg_float, None, norm_alpha, norm_beta, norm_type)

        if output_dtype == "uint8":
            return np.clip(avg_float, 0, 255).astype(np.uint8)

        return avg_float.astype(np.float32)

    finally:
        cap.release()
        if show:
            cv.destroyAllWindows()
        if verbose:
            print(f"[recursive_temporal_average] Time: {time.perf_counter() - start_time:.6f}s")

    