import numpy as np
from thermal.noise_reduction import thermal_bilateral_filter

def test_bilateral_shape():
    img = np.random.randint(0,255,(64,64),dtype=np.uint8)
    out = thermal_bilateral_filter(img)
    assert out.shape == img.shape
