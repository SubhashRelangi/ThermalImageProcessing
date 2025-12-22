import cv2
import numpy as np
import time

def thermal_nlm_denoise(
    src: np.ndarray,
    *,
    h: float = 3.5,
    template_window: int = 7,
    search_window: int = 15
) -> np.ndarray:
    
    """
    Apply Non-Local Means (NLM) denoising to a single-channel thermal image.

    This function reduces random thermal noise by exploiting self-similarity
    across the image while preserving important thermal structures such as
    edges, gradients, and defect signatures. It is designed specifically for
    spatial denoising of single-frame thermal imagery.

    -------------------------------------------------------------------------
    PARAMETERS
    -------------------------------------------------------------------------

    src : np.ndarray
        Description:
            Input thermal image. Must be a single-channel (grayscale) image.
        Min & Max values:
            Shape: (H, W) or (H, W, 1)
            Dtype: uint8, uint16, float32, float64
        Units:
            Pixel intensity (sensor-dependent thermal units)
        Default values:
            Required (no default)
        Best case values:
            Cleanly captured single-channel thermal frame with stable noise
            characteristics.

    h : float
        Description:
            Filtering strength controlling the degree of denoising.
            Larger values increase smoothing but may suppress fine thermal details.
        Min & Max values:
            > 0.0 (practically 1.0 – 10.0 for thermal images)
        Units:
            Intensity similarity threshold (unitless, algorithmic parameter)
        Default values:
            3.5
        Best case values:
            2.5 – 4.0 for analysis-grade thermal denoising;
            higher values only for visualization.

    template_window : int
        Description:
            Size of the local patch used to compare image similarity.
        Min & Max values:
            Odd integers ≥ 3
        Units:
            Pixels
        Default values:
            7
        Best case values:
            7 for most thermal images (good balance of locality and stability).

    search_window : int
        Description:
            Size of the search region used to find similar patches.
        Min & Max values:
            Odd integers ≥ template_window
        Units:
            Pixels
        Default values:
            15
        Best case values:
            15–21 for thermal imagery; larger values increase robustness but
            also computational cost.

    preserve_depth : bool
        Description:
            Indicates whether the function should attempt to preserve the
            original image depth and dynamic range.
            (Note: current implementation returns an 8-bit denoised image.)
        Min & Max values:
            True or False
        Units:
            Boolean flag
        Default values:
            True
        Best case values:
            True for consistent downstream thermal processing pipelines.

    -------------------------------------------------------------------------
    RETURNS
    -------------------------------------------------------------------------

    out_image : np.ndarray
        Description:
            Denoised thermal image.
        Min & Max values:
            uint8 image with range [0, 255]
        Units:
            Pixel intensity
        Shape:
            Same spatial dimensions as input image.

    -------------------------------------------------------------------------
    NOTES
    -------------------------------------------------------------------------
    - This implementation uses OpenCV's fastNlMeansDenoising, which operates
      internally on 8-bit single-channel images.
    - Higher `h` values increase noise removal but can reduce thermal detail.
    - For defect detection and quantitative thermal analysis, moderate denoising
      (not aggressive smoothing) is recommended.
    - For significant SNR improvement, temporal NLM (multi-frame) is preferred
      over spatial NLM.

    -------------------------------------------------------------------------
    """

    start_time = time.time()

    try:
        # ---------- Validate input ----------
        if src is None:
            raise ValueError("Input image is None")

        if not isinstance(src, np.ndarray):
            raise TypeError("Input must be numpy ndarray")

        if src.ndim == 2:
            img = src
        elif src.ndim == 3 and src.shape[2] == 1:
            img = src[..., 0]
        else:
            raise ValueError(
                f"NLM expects SINGLE-CHANNEL image, got shape {src.shape}"
            )

        if h <= 0:
            raise ValueError("h must be > 0")

        if template_window < 3 or template_window % 2 == 0:
            raise ValueError("template_window must be odd and >= 3")

        if search_window < template_window or search_window % 2 == 0:
            raise ValueError(
                "search_window must be odd and >= template_window"
            )

        # ---------- Normalize to uint8 ----------
        if img.dtype == np.uint8:
            img_u8 = img.copy()

        elif np.issubdtype(img.dtype, np.integer):
            img_u8 = cv2.normalize(
                img, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

        elif np.issubdtype(img.dtype, np.floating):
            mn, mx = img.min(), img.max()
            if mx == mn:
                raise ValueError("Flat image: cannot denoise")
            img_u8 = ((img - mn) / (mx - mn) * 255).astype(np.uint8)

        else:
            raise TypeError(f"Unsupported dtype: {img.dtype}")

        # ---------- NLM ----------
        denoised = cv2.fastNlMeansDenoising(
            img_u8,
            None,
            h=float(h),
            templateWindowSize=int(template_window),
            searchWindowSize=int(search_window)
        )

        return denoised

    except Exception as e:
        raise RuntimeError(f"[thermal_nlm_denoise] {e}") from e

    finally:
        print(f"[thermal_nlm_denoise] Time: {time.time() - start_time:.4f}s")

