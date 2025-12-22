import numpy as np
from thermal.noise_reduction import thermal_nlm_denoise

def test_nlm_shape():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    out = thermal_nlm_denoise(img)
    assert out.shape == img.shape
