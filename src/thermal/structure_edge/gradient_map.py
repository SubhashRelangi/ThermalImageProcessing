import cv2 as cv
import numpy as np
import time
from typing import Union

ImageArray = np.ndarray

def gradient_map(
    gray_image: Union[str, ImageArray],
    *,
    dx: int = 1,
    dy: int = 1,
    ddepth: int = cv.CV_32F,
    ksize: int = 3,
    sobel_scale: float = 0.8,
    sobel_delta: float = 0.0,
    sobel_border: int = cv.BORDER_DEFAULT,
    normalize_output: bool = True,
    colormap: int = cv.COLORMAP_JET,
) -> ImageArray:
   
    """
    Compute a Sobel-based gradient magnitude map and visualize it as a colorized
    gradient image.

    This function computes horizontal and vertical Sobel gradients,
    combines them into a gradient magnitude map, optionally normalizes
    the result, and applies a color map for visualization.

    ---------------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------------

    gray_image : str | np.ndarray
        Description:
            Input image for gradient computation. Can be either a file path
            or a preloaded NumPy array. The image must be single-channel
            (grayscale).

        Min & Max values:
            - Shape: (H, W)
            - Dtype:
                uint8   : [0, 255]
                uint16  : [0, 65535]
                float32 : arbitrary range (internally converted)

        Units:
            Pixel intensity

        Default:
            Required

        Best-case values:
            Noise-reduced grayscale image with good contrast.

    ---------------------------------------------------------------------

    dx : int
        Description:
            Order of the derivative in the x-direction for Sobel filtering.

        Min & Max values:
            Typically 0 or 1

        Units:
            Derivative order

        Default:
            1

        Best-case values:
            1 for detecting vertical edges.

    ---------------------------------------------------------------------

    dy : int
        Description:
            Order of the derivative in the y-direction for Sobel filtering.

        Min & Max values:
            Typically 0 or 1

        Units:
            Derivative order

        Default:
            1

        Best-case values:
            1 for detecting horizontal edges.

    ---------------------------------------------------------------------

    ddepth : int
        Description:
            Desired depth of the output Sobel gradients.

        Min & Max values:
            OpenCV depth enums (e.g., cv.CV_16S, cv.CV_32F, cv.CV_64F)

        Units:
            OpenCV data type

        Default:
            cv.CV_32F

        Best-case values:
            cv.CV_32F for numerical stability and precision.

    ---------------------------------------------------------------------

    ksize : int
        Description:
            Size of the Sobel kernel. Must be an odd integer.

        Min & Max values:
            Odd integers ≥ 1 (commonly 3, 5, 7)

        Units:
            Pixels

        Default:
            3

        Best-case values:
            3 for fine edges, 5–7 for smoother gradient maps.

    ---------------------------------------------------------------------

    sobel_scale : float
        Description:
            Scaling factor applied to the computed Sobel derivatives.
            Used to control gradient magnitude amplification.

        Min & Max values:
            > 0.0

        Units:
            Unitless scale factor

        Default:
            0.8

        Best-case values:
            0.5 – 1.5 depending on image contrast.

    ---------------------------------------------------------------------

    sobel_delta : float
        Description:
            Value added to the Sobel results before further processing.

        Min & Max values:
            No hard limit

        Units:
            Pixel intensity offset

        Default:
            0.0

        Best-case values:
            0.0 for standard gradient computation.

    ---------------------------------------------------------------------

    sobel_border : int
        Description:
            Pixel extrapolation method used at image borders.

        Allowed values:
            OpenCV border types such as:
                cv.BORDER_DEFAULT
                cv.BORDER_REFLECT
                cv.BORDER_REPLICATE

        Units:
            OpenCV enum

        Default:
            cv.BORDER_DEFAULT

        Best-case values:
            cv.BORDER_DEFAULT for most thermal and natural images.

    ---------------------------------------------------------------------

    normalize_output : bool
        Description:
            Whether to normalize the gradient magnitude to the range [0, 255]
            before visualization.

        Min & Max values:
            True or False

        Units:
            Boolean flag

        Default:
            True

        Best-case values:
            True for visualization, False for quantitative analysis.

    ---------------------------------------------------------------------

    colormap : int
        Description:
            OpenCV colormap applied to the gradient magnitude image
            for visualization.

        Allowed values:
            cv.COLORMAP_JET
            cv.COLORMAP_INFERNO
            cv.COLORMAP_TURBO
            cv.COLORMAP_HOT
            etc.

        Units:
            OpenCV enum

        Default:
            cv.COLORMAP_JET

        Best-case values:
            cv.COLORMAP_INFERNO or cv.COLORMAP_TURBO for thermal-style gradients.

    ---------------------------------------------------------------------
    RETURNS
    ---------------------------------------------------------------------

    out : np.ndarray
        Description:
            Colorized gradient magnitude image.

        Shape:
            (H, W, 3)

        Dtype:
            uint8

        Units:
            Pseudo-color intensity

    ---------------------------------------------------------------------
    EXCEPTIONS
    ---------------------------------------------------------------------

    Raises RuntimeError if:
        - Image loading fails
        - Input validation fails
        - Sobel computation fails

    ---------------------------------------------------------------------
    USE CASES
    ---------------------------------------------------------------------

    - Edge and structure visualization
    - Feature exploration for ML preprocessing
    - Thermal-style gradient inspection
    - Debugging spatial detail retention

    ---------------------------------------------------------------------
    """


    start_time = time.time()

    try:
        # ---------- load ----------
        if isinstance(gray_image, str):
            img = cv.imread(gray_image, cv.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(gray_image)
        else:
            img = gray_image

        # ---------- validation ----------
        if img is None:
            raise ValueError("Input image is None")
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image must be numpy ndarray")
        if img.ndim != 2:
            raise ValueError("Input image must be single-channel")
        if img.size == 0:
            raise ValueError("Input image is empty")

        if ksize < 1 or ksize % 2 == 0:
            raise ValueError("ksize must be odd >= 1")
        if sobel_scale <= 0:
            raise ValueError("sobel_scale must be > 0")

        img_f = img.astype(np.float32)

        # ---------- sobel ----------
        grad_x = cv.Sobel(
            img_f, ddepth, dx, 0,
            ksize=ksize,
            scale=sobel_scale,
            delta=sobel_delta,
            borderType=sobel_border
        )

        grad_y = cv.Sobel(
            img_f, ddepth, 0, dy,
            ksize=ksize,
            scale=sobel_scale,
            delta=sobel_delta,
            borderType=sobel_border
        )

        magnitude = cv.magnitude(grad_x, grad_y)

        # ---------- normalize ----------
        if normalize_output:
            magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        mag_u8 = np.clip(magnitude, 0, 255).astype(np.uint8)
        out = cv.applyColorMap(mag_u8, colormap)

        print(f"[gradient_map] Time: {time.time() - start_time:.6f}s")
        return out

    except Exception as e:
        raise RuntimeError(f"gradient_map failed: {e}")

