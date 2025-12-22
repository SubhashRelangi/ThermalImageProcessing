import time
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet

def wavelet_denoise_thermal(
    image,
    *,
    method: str = "BayesShrink",
    mode: str = "soft",
    wavelet: str = "db4",
    wavelet_levels: int = 2,
    rescale_sigma: bool = True,
    preserve_dtype: bool = True
) -> np.ndarray:
    
    """
    Apply wavelet-based denoising to a single-channel thermal image.

    This function performs spatial wavelet shrinkage to suppress
    random thermal noise while preserving important thermal structures
    such as edges, gradients, and localized heat patterns. It is
    well-suited for thermal imagery where noise is largely additive
    and multi-scale in nature.

    ---------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------

    image : np.ndarray | str
        Description:
            Input thermal image. Can be either:
            - A NumPy array already loaded in memory, or
            - A filesystem path to a grayscale thermal image.

            The image must be single-channel (grayscale). Multi-channel
            inputs are not supported.

        Min & Max values:
            Depends on dtype:
                uint8  : [0, 255]
                uint16 : [0, 65535]
                float  : [0.0, 1.0]

        Units:
            Thermal intensity (raw sensor units or normalized radiance)

        Default:
            None (required)

        Best-case values:
            Radiometrically stable thermal frames with stationary noise
            characteristics.


    method : str
        Description:
            Wavelet threshold selection strategy used to estimate
            noise statistics and compute shrinkage thresholds.

            Passed directly to `skimage.restoration.denoise_wavelet`.

        Allowed values:
            - "BayesShrink"
            - "VisuShrink"

        Units:
            Algorithmic mode (string identifier)

        Default:
            "BayesShrink"

        Best-case values:
            "BayesShrink" for adaptive, data-driven thermal noise
            suppression.


    mode : str
        Description:
            Type of thresholding applied to wavelet coefficients.

            Determines how coefficients below the threshold are treated.

        Allowed values:
            - "soft"
            - "hard"

        Units:
            Thresholding strategy

        Default:
            "soft"

        Best-case values:
            "soft" for thermal images to avoid ringing artifacts
            and preserve smooth temperature transitions.


    wavelet : str
        Description:
            Wavelet basis used for multi-resolution decomposition.

            Controls spatial-frequency sensitivity and edge behavior.

        Allowed values:
            Any wavelet supported by PyWavelets (e.g., "db4", "haar",
            "sym4", "coif1").

        Units:
            Wavelet family identifier

        Default:
            "db4"

        Best-case values:
            "db4" or similar Daubechies wavelets for balanced
            edge preservation and noise suppression.


    wavelet_levels : int
        Description:
            Number of decomposition levels used in the wavelet transform.

            Higher levels capture coarser structures but increase
            smoothing and computational cost. The actual level used
            is automatically clamped based on image size.

        Min & Max values:
            Min: 1
            Max: floor(log2(min(image height, image width)))

        Units:
            Decomposition levels

        Default:
            2

        Best-case values:
            2 – 3 for typical thermal imagery;
            higher values only for very large images.


    rescale_sigma : bool
        Description:
            Whether to rescale the noise standard deviation estimate
            for each wavelet sub-band.

            Improves robustness when noise characteristics vary
            across scales.

        Min & Max values:
            True / False

        Units:
            Boolean flag

        Default:
            True

        Best-case values:
            True for real thermal sensors with scale-dependent noise.


    preserve_dtype : bool
        Description:
            Whether to restore the output image to the original
            input data type after denoising.

            If False, the output is returned as a float32 image
            in the range [0.0, 1.0].

        Min & Max values:
            True / False

        Units:
            Boolean flag

        Default:
            True

        Best-case values:
            True for consistency in downstream thermal pipelines;
            False for further floating-point analysis.

    ---------------------------------------------------------------------
    RETURNS
    ---------------------------------------------------------------------

    np.ndarray
        Description:
            Wavelet-denoised thermal image.

        Shape:
            Same spatial dimensions as input image.

        Min & Max values:
            If preserve_dtype=True:
                uint8  : [0, 255]
                uint16 : [0, 65535]
            If preserve_dtype=False:
                float32 : [0.0, 1.0]

        Units:
            Thermal intensity (preserved or normalized)

    ---------------------------------------------------------------------
    EXCEPTIONS
    ---------------------------------------------------------------------

    Raises RuntimeError if:
        - Input validation fails
        - Unsupported image dtype is provided
        - Image is not single-channel
        - Wavelet denoising fails internally
    """


    start = time.perf_counter()

    try:
        # -------- Load --------
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("Input must be numpy array or file path")

        # -------- Validate --------
        if img.ndim == 2:
            pass
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img[..., 0]
        else:
            raise ValueError("Expected SINGLE-CHANNEL thermal image")

        orig_dtype = img.dtype

        # -------- Normalize --------
        if orig_dtype == np.uint8:
            img_f = img.astype(np.float32) / 255.0
        elif orig_dtype == np.uint16:
            img_f = img.astype(np.float32) / 65535.0
        elif np.issubdtype(orig_dtype, np.floating):
            img_f = np.clip(img.astype(np.float32), 0.0, 1.0)
        else:
            raise TypeError(f"Unsupported dtype: {orig_dtype}")

        # -------- Wavelet level safety --------
        h, w = img_f.shape
        max_levels = int(np.floor(np.log2(min(h, w))))
        wavelet_levels = max(1, min(wavelet_levels, max_levels))

        # -------- Denoise --------
        den_f = denoise_wavelet(
            img_f,
            method=method,
            mode=mode,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
            rescale_sigma=rescale_sigma,
            channel_axis=None
        )

        # -------- Restore dtype --------
        if preserve_dtype:
            if orig_dtype == np.uint16:
                out = (den_f * 65535).clip(0, 65535).astype(np.uint16)
            else:
                out = (den_f * 255).clip(0, 255).astype(np.uint8)
        else:
            out = den_f.astype(np.float32)

    except Exception as e:
        raise RuntimeError(f"[Wavelet Denoising Failed] {e}") from e

    finally:
        elapsed = time.perf_counter() - start
        print(f"[Wavelet Denoise] Time: {elapsed:.4f}s")

    return out

