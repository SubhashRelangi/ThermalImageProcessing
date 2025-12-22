import numpy as np
from thermal.noise_reduction import thermal_median_filter

def test_median_shape():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    out = thermal_median_filter(img, 3)
    assert out.shape == img.shape
