"""Image operations for the Image lab (numpy and scipy.ndimage). Images are float32 arrays in
[0, 1], shaped (H, W) for grey or (H, W, 3) for colour; every operation returns a new array in
the same range."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "adaptive_threshold", "box_blur", "brightness_contrast", "canny", "convolve", "equalize",
    "fft_magnitude", "flip", "frequency_filter", "gamma", "gaussian_blur", "grey", "harris_corners",
    "invert", "kmeans_colours", "laplacian", "median", "morphology", "noise_gaussian",
    "noise_salt_pepper", "otsu_level", "rotate", "scale", "sharpen", "sobel", "threshold",
]


def _clip(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _per_channel(image: np.ndarray, fn) -> np.ndarray:
    if image.ndim == 2:
        return _clip(fn(image))
    return _clip(np.stack([fn(image[..., c]) for c in range(image.shape[2])], axis=-1))


def grey(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return (image[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)


def _normalise(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values, dtype=np.float32) if high - low < 1e-12 else \
        ((values - low) / (high - low)).astype(np.float32)


# Smoothing and sharpening ---------------------------------------------------------------------------
def gaussian_blur(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return _per_channel(image, lambda c: ndimage.gaussian_filter(c, sigma))


def box_blur(image: np.ndarray, size: int = 5) -> np.ndarray:
    return _per_channel(image, lambda c: ndimage.uniform_filter(c, size))


def median(image: np.ndarray, size: int = 3) -> np.ndarray:
    return _per_channel(image, lambda c: ndimage.median_filter(c, size))


def sharpen(image: np.ndarray, amount: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Unsharp masking: add back the detail a blur removes."""
    return _per_channel(image, lambda c: c + amount * (c - ndimage.gaussian_filter(c, sigma)))


# Edges, corners and thresholds -----------------------------------------------------------------------
def sobel(image: np.ndarray) -> np.ndarray:
    g = grey(image)
    return _normalise(np.hypot(ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0)))


def laplacian(image: np.ndarray) -> np.ndarray:
    return _normalise(np.abs(ndimage.laplace(ndimage.gaussian_filter(grey(image), 1.0))))


def canny(image: np.ndarray, sigma: float = 1.4, low: float = 0.1, high: float = 0.25) -> np.ndarray:
    """Canny's detector: smooth, gradient, thin to one pixel (non-maximum suppression), then keep
    weak edges only where they connect to strong ones (hysteresis)."""
    g = ndimage.gaussian_filter(grey(image), sigma)
    gx, gy = ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0)
    magnitude = np.hypot(gx, gy)
    magnitude = magnitude / (magnitude.max() or 1.0)
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    sector = (np.round(angle / 45.0).astype(int)) % 4  # 0: horizontal, 1: 45°, 2: vertical, 3: 135°
    padded = np.pad(magnitude, 1)
    h, w = magnitude.shape
    neighbours = {0: ((0, 1), (0, -1)), 1: ((-1, 1), (1, -1)), 2: ((-1, 0), (1, 0)), 3: ((-1, -1), (1, 1))}
    keep = np.zeros_like(magnitude, dtype=bool)
    for code, ((dy1, dx1), (dy2, dx2)) in neighbours.items():
        first = padded[1 + dy1:1 + dy1 + h, 1 + dx1:1 + dx1 + w]
        second = padded[1 + dy2:1 + dy2 + h, 1 + dx2:1 + dx2 + w]
        keep |= (sector == code) & (magnitude >= first) & (magnitude >= second)
    thin = magnitude * keep
    strong, weak = thin >= high, thin >= low
    labels, count = ndimage.label(weak, structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros_like(magnitude, dtype=np.float32)
    connected = np.zeros(count + 1, dtype=bool)
    connected[np.unique(labels[strong])] = True
    connected[0] = False
    return connected[labels].astype(np.float32)


def harris_corners(image: np.ndarray, sigma: float = 1.5, k: float = 0.05,
                   threshold: float = 0.05) -> np.ndarray:
    """Harris corners marked in red on the image (grey shown as colour)."""
    g = grey(image)
    gx, gy = ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0)
    sxx, syy, sxy = (ndimage.gaussian_filter(v, sigma) for v in (gx * gx, gy * gy, gx * gy))
    response = sxx * syy - sxy ** 2 - k * (sxx + syy) ** 2
    peaks = (response == ndimage.maximum_filter(response, size=7)) & (response > threshold * response.max())
    base = np.repeat(g[..., None], 3, axis=2) if image.ndim == 2 else image[..., :3].copy()
    marks = ndimage.binary_dilation(peaks, iterations=2)
    base[marks] = (0.9, 0.25, 0.2)
    return _clip(base)


def otsu_level(image: np.ndarray) -> float:
    """The threshold that best separates the two halves of the histogram (Otsu's method)."""
    hist, edges = np.histogram(grey(image), bins=256, range=(0.0, 1.0))
    weights = hist / max(hist.sum(), 1)
    centres = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(weights)
    mu = np.cumsum(weights * centres)
    between = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    return float(centres[int(np.argmax(between))])


def threshold(image: np.ndarray, level: float = 0.5, method: str = "fixed") -> np.ndarray:
    g = grey(image)
    cut = otsu_level(g) if method == "otsu" else level
    return (g > cut).astype(np.float32)


def adaptive_threshold(image: np.ndarray, size: int = 25, offset: float = 0.02) -> np.ndarray:
    g = grey(image)
    return (g > ndimage.uniform_filter(g, size) - offset).astype(np.float32)


def morphology(image: np.ndarray, operation: str = "open", size: int = 3) -> np.ndarray:
    footprint = np.ones((size, size), dtype=bool)
    fn = {"erode": ndimage.grey_erosion, "dilate": ndimage.grey_dilation,
          "open": ndimage.grey_opening, "close": ndimage.grey_closing}[operation]
    return _per_channel(image, lambda c: fn(c, footprint=footprint))


# Tone and colour -------------------------------------------------------------------------------------
def equalize(image: np.ndarray) -> np.ndarray:
    def one(channel: np.ndarray) -> np.ndarray:
        hist, edges = np.histogram(channel, bins=256, range=(0.0, 1.0))
        cdf = np.cumsum(hist).astype(np.float64)
        cdf = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), 1)
        return np.interp(channel, edges[:-1], cdf)
    return _per_channel(image, one)


def gamma(image: np.ndarray, value: float = 0.7) -> np.ndarray:
    return _clip(np.power(np.clip(image, 0, 1), value))


def brightness_contrast(image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    return _clip((image - 0.5) * contrast + 0.5 + brightness)


def invert(image: np.ndarray) -> np.ndarray:
    return _clip(1.0 - image)


def kmeans_colours(image: np.ndarray, k: int = 4, seed: int = 0, iterations: int = 12) -> np.ndarray:
    """Colour quantisation: every pixel takes the nearest of ``k`` colours found by k-means."""
    pixels = image.reshape(-1, image.shape[2] if image.ndim == 3 else 1).astype(np.float64)
    rng = np.random.default_rng(seed)
    sample = pixels[rng.choice(len(pixels), size=min(len(pixels), 4000), replace=False)]
    centres = sample[rng.choice(len(sample), size=min(k, len(sample)), replace=False)]
    for _ in range(iterations):
        nearest = np.argmin(((sample[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        for j in range(len(centres)):
            members = sample[nearest == j]
            if len(members):
                centres[j] = members.mean(axis=0)
    labels = np.argmin(((pixels[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
    return _clip(centres[labels].reshape(image.shape))


# Noise ------------------------------------------------------------------------------------------------
def noise_gaussian(image: np.ndarray, sigma: float = 0.08, seed: int = 0) -> np.ndarray:
    return _clip(image + np.random.default_rng(seed).normal(0.0, sigma, image.shape))


def noise_salt_pepper(image: np.ndarray, amount: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy()
    mask = rng.random(image.shape[:2])
    out[mask < amount / 2] = 0.0
    out[mask > 1 - amount / 2] = 1.0
    return _clip(out)


# Geometry and frequency ------------------------------------------------------------------------------
def rotate(image: np.ndarray, angle: float = 15.0) -> np.ndarray:
    axes = (1, 0)
    return _clip(ndimage.rotate(image, angle, axes=axes, reshape=False, order=1, mode="nearest"))


def scale(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    zoom = (factor, factor) + ((1,) if image.ndim == 3 else ())
    return _clip(ndimage.zoom(image, zoom, order=1))


def flip(image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    return _clip(image[:, ::-1] if direction == "horizontal" else image[::-1])


def convolve(image: np.ndarray, kernel: tuple[float, ...] = (0, -1, 0, -1, 5, -1, 0, -1, 0)) -> np.ndarray:
    """A 3×3 kernel of your own (row by row)."""
    weights = np.asarray(kernel, dtype=np.float32).reshape(3, 3)
    return _per_channel(image, lambda c: ndimage.convolve(c, weights, mode="nearest"))


def fft_magnitude(image: np.ndarray) -> np.ndarray:
    """The log magnitude of the 2-D Fourier transform, zero frequency in the middle."""
    spectrum = np.fft.fftshift(np.fft.fft2(grey(image)))
    return _normalise(np.log1p(np.abs(spectrum)))


def frequency_filter(image: np.ndarray, kind: str = "low", cutoff: float = 0.1) -> np.ndarray:
    """A Gaussian low- or high-pass filter in the frequency domain (cutoff as a fraction of the
    sampling frequency)."""
    def one(channel: np.ndarray) -> np.ndarray:
        fy = np.fft.fftfreq(channel.shape[0])[:, None]
        fx = np.fft.fftfreq(channel.shape[1])[None, :]
        low = np.exp(-(fx ** 2 + fy ** 2) / (2 * max(cutoff, 1e-4) ** 2))
        mask = low if kind == "low" else 1.0 - low
        filtered = np.real(np.fft.ifft2(np.fft.fft2(channel) * mask))
        return filtered if kind == "low" else _normalise(filtered)
    return _per_channel(image, one)
