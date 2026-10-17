"""PSNR и SSIM по маске — общие для eval_views.py и eval_novel_views.py (картинки 0..1, HxWx3)."""

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter


def psnr(a, b, m):
    mse = np.mean((a[m] - b[m]) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim(a, b, m, sigma=1.5):
    """SSIM по яркости, окно Гаусса; среднее по маске, сжатой на радиус окна."""
    ya, yb = a @ [0.299, 0.587, 0.114], b @ [0.299, 0.587, 0.114]
    mu_a, mu_b = gaussian_filter(ya, sigma), gaussian_filter(yb, sigma)
    va = gaussian_filter(ya * ya, sigma) - mu_a ** 2
    vb = gaussian_filter(yb * yb, sigma) - mu_b ** 2
    cov = gaussian_filter(ya * yb, sigma) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s[binary_erosion(m, iterations=int(3 * sigma))].mean())
