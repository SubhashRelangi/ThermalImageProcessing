import numpy as np
from thermal.noise_reduction.wavelet import wavelet_denoise_thermal

def test_wavelet_shape():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    out = wavelet_denoise_thermal(img)
    assert out.shape == img.shape
