from skimage.restoration import (denoise_wavelet, estimate_sigma)


def wt(noisy, std):
    return denoise_wavelet(noisy, sigma=std)
