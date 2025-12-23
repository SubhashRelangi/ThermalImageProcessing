from typing import Tuple
import cv2 as cv
import numpy as np
import time

def HighPassEdge(
    image: np.ndarray,
    *,
    filter_mode: str = "highpass",
    radius: int = 50,
    gain: float = 1.5,
    sigma: float = 20.0,
    order: int = 2,
    fft_flag: int = cv.DFT_COMPLEX_OUTPUT,
) -> np.ndarray:
    
    """
    Perform frequency-domain edge enhancement using configurable
    high-pass, low-pass, band-pass, Gaussian, or Butterworth filters.

    This function applies FFT-based filtering to a single-channel image
    to selectively enhance high-frequency components (edges and fine
    structures) while optionally retaining low-frequency content.
    It is designed for structure enhancement in thermal and grayscale
    imagery where edge clarity is critical.

    ---------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------

    image : np.ndarray
        Description:
            Input image to be processed. Must be a single-channel
            (grayscale or thermal) image.

        Min & Max values:
            uint8  : [0, 255]
            uint16 : [0, 65535]
            float  : unrestricted (will be internally converted)

        Units:
            Pixel intensity (radiometric or relative)

        Default:
            None (required)

        Best-case values:
            Pre-denoised or lightly smoothed thermal images for
            stable frequency response.

    ---------------------------------------------------------------------

    filter_mode : str
        Description:
            Type of frequency-domain filter to apply.

        Supported values:
            - "lowpass"        : Ideal low-pass filter
            - "highpass"       : Ideal high-pass filter
            - "bandpass"       : Ideal band-pass filter
            - "gaussian_lp"    : Gaussian low-pass filter
            - "gaussian_hp"    : Gaussian high-pass filter
            - "butterworth_lp" : Butterworth low-pass filter
            - "butterworth_hp" : Butterworth high-pass filter

        Default:
            "highpass"

        Best-case values:
            - "gaussian_hp" or "butterworth_hp" for smooth edge enhancement
            - "highpass" for aggressive sharpening

    ---------------------------------------------------------------------

    radius : int
        Description:
            Cutoff radius in the frequency domain that determines
            which frequencies are preserved or suppressed.

        Min & Max values:
            Min: 1
            Max: min(image width, image height) / 2

        Units:
            Frequency pixels

        Default:
            50

        Best-case values:
            20 – 60 depending on image resolution and edge scale.

    ---------------------------------------------------------------------

    gain : float
        Description:
            Amplification factor applied to high-frequency components.
            Controls edge strength after filtering.

        Min & Max values:
            Min: 1.0
            Max: No hard limit (practically ≤ 2.0)

        Units:
            Unitless scaling factor

        Default:
            1.5

        Best-case values:
            1.2 – 1.6 for thermal images to avoid ringing artifacts.

    ---------------------------------------------------------------------

    sigma : float
        Description:
            Standard deviation for Gaussian frequency-domain filters.
            Used only when filter_mode is "gaussian_lp" or "gaussian_hp".

        Min & Max values:
            Min: > 0
            Max: No hard limit (image-dependent)

        Units:
            Frequency-domain spread

        Default:
            20.0

        Best-case values:
            15 – 30 for smooth edge emphasis without harsh transitions.

    ---------------------------------------------------------------------

    order : int
        Description:
            Order of the Butterworth filter. Higher values result in
            steeper frequency roll-off.

        Min & Max values:
            Min: 1
            Max: Practically ≤ 5

        Units:
            Unitless (filter order)

        Default:
            2

        Best-case values:
            2 or 3 for stable Butterworth response.

    ---------------------------------------------------------------------

    fft_flag : int
        Description:
            OpenCV DFT flag specifying output format for the Fourier
            transform.

        Min & Max values:
            OpenCV-supported DFT flags

        Units:
            OpenCV enum

        Default:
            cv.DFT_COMPLEX_OUTPUT

        Best-case values:
            cv.DFT_COMPLEX_OUTPUT (required for inverse DFT magnitude)

    ---------------------------------------------------------------------
    RETURNS
    ---------------------------------------------------------------------

    np.ndarray
        Description:
            High-pass enhanced image reconstructed from frequency domain.

        Dtype:
            uint8

        Min & Max values:
            [0, 255]

        Units:
            Pixel intensity

        Shape:
            Same spatial dimensions as input image.

    ---------------------------------------------------------------------
    NOTES
    ---------------------------------------------------------------------
    - Input image must be single-channel.
    - Internally converts image to float32 for FFT stability.
    - High gain or sharp cutoffs may introduce ringing artifacts.
    - Gaussian and Butterworth filters are preferred over ideal filters
      for thermal imagery.

    ---------------------------------------------------------------------
    TYPICAL USE CASES
    ---------------------------------------------------------------------
    - Thermal edge enhancement
    - Structural feature amplification
    - Preprocessing for defect detection
    - Enhancing contours before segmentation or tracking

    ---------------------------------------------------------------------
    EXCEPTIONS
    ---------------------------------------------------------------------
    Raises RuntimeError or ValueError on invalid input parameters,
    unsupported filter modes, or FFT processing failures.
    """

    start_time = time.time()

    if image is None:
        raise ValueError("Input image is None")
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be numpy ndarray")
    if image.ndim != 2:
        raise ValueError("Input image must be single-channel")
    if image.size == 0:
        raise ValueError("Input image is empty")
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if gain < 1.0:
        raise ValueError("gain must be >= 1.0")

    valid_modes = {
        "lowpass", "highpass", "bandpass",
        "gaussian_lp", "gaussian_hp",
        "butterworth_lp", "butterworth_hp"
    }
    if filter_mode not in valid_modes:
        raise ValueError(f"Unsupported filter_mode: {filter_mode}")

    img_f = np.float32(image)
    F = cv.dft(img_f, flags=fft_flag)
    F_shift = np.fft.fftshift(F)

    h, w = img_f.shape
    cx, cy = w // 2, h // 2
    U, V = np.meshgrid(np.arange(w), np.arange(h))
    D = np.sqrt((U - cx) ** 2 + (V - cy) ** 2)

    mask = np.zeros((h, w), np.float32)

    if filter_mode == "lowpass":
        mask[D <= radius] = 1.0
    elif filter_mode == "highpass":
        mask[D > radius] = 1.0
    elif filter_mode == "bandpass":
        r1, r2 = max(1, radius // 2), radius
        if r1 >= r2:
            raise ValueError("Invalid bandpass radii")
        mask[(D >= r1) & (D <= r2)] = 1.0
    elif filter_mode == "gaussian_lp":
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        mask = np.exp(-(D ** 2) / (2 * sigma ** 2))
    elif filter_mode == "gaussian_hp":
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        mask = 1.0 - np.exp(-(D ** 2) / (2 * sigma ** 2))
    elif filter_mode == "butterworth_lp":
        if order < 1:
            raise ValueError("order must be >= 1")
        mask = 1.0 / (1.0 + (D / radius) ** (2 * order))
    elif filter_mode == "butterworth_hp":
        if order < 1:
            raise ValueError("order must be >= 1")
        mask = 1.0 - (1.0 / (1.0 + (D / radius) ** (2 * order)))

    mask2 = cv.merge([mask, mask])
    HF = F_shift * mask2 * gain
    LF = F_shift * (1.0 - mask2)
    merged = HF + LF

    spatial = cv.idft(np.fft.fftshift(merged))
    mag = cv.magnitude(spatial[:, :, 0], spatial[:, :, 1])

    out_img = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)

    print(f"[HighPassEdge] Time: {time.time() - start_time:.6f}s")
    return out_img

