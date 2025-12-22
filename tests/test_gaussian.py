import numpy as np
from thermal.noise_reduction import thermal_gaussian_filter

def test_gaussian_shape():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    out = thermal_gaussian_filter(img)
    assert out.shape == img.shape
