"""Audio signal processing with numpy and scipy only: WAV files, resampling, test signals, the
short-time Fourier transform, mel filterbanks, MFCCs, filters and a pitch estimate. Signals are
float arrays in −1..1, mono.

Mel scale: ``m = 2595 log10(1 + f / 700)`` (the HTK formula); the filterbank has triangular
filters spaced evenly on it, area-normalised. MFCCs are the type-II DCT (orthonormal) of the
log mel energies.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np

__all__ = ["GENERATORS", "bandpass", "envelope", "generate", "hz_to_mel", "load_wav", "log_mel_image",
           "mel_filterbank", "mel_spectrogram", "mel_to_hz", "mfcc", "pitch", "resample", "save_wav",
           "spectrogram", "stft"]

GENERATORS = {"sine": "Sine tone", "chord": "Chord (three tones)", "chirp": "Chirp (sweep)",
              "square": "Square wave", "noise": "White noise", "pink": "Pink noise", "beeps": "Beeps"}


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """(samples in −1..1, mono; sample rate). 8/16/32-bit integer and float WAV files."""
    from scipy.io import wavfile

    rate, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.dtype.kind == "u":
        signal = (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype.kind == "i":
        signal = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        signal = data.astype(np.float64)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return np.clip(signal, -1.0, 1.0), int(rate)


def save_wav(path: str | Path, signal: np.ndarray, rate: int) -> Path:
    from scipy.io import wavfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), int(rate), (np.clip(signal, -1, 1) * 32767).astype(np.int16))
    return path


def resample(signal: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target or not len(signal):
        return np.asarray(signal, dtype=np.float64)
    from scipy.signal import resample_poly

    g = gcd(int(rate), int(target))
    return resample_poly(signal, target // g, rate // g)


def generate(kind: str, seconds: float = 2.0, rate: int = 16_000, frequency: float = 440.0,
             end_frequency: float = 2000.0, amplitude: float = 0.5, seed: int = 0) -> np.ndarray:
    """A test signal (see :data:`GENERATORS`)."""
    t = np.arange(int(seconds * rate)) / rate
    if kind == "sine":
        wave = np.sin(2 * np.pi * frequency * t)
    elif kind == "chord":  # a major triad
        wave = sum(np.sin(2 * np.pi * frequency * r * t) for r in (1.0, 1.25, 1.5)) / 3
    elif kind == "chirp":
        from scipy.signal import chirp

        wave = chirp(t, frequency, seconds, end_frequency, method="logarithmic")
    elif kind == "square":
        wave = np.sign(np.sin(2 * np.pi * frequency * t))
    elif kind in ("noise", "pink"):
        white = np.random.default_rng(seed).standard_normal(len(t))
        if kind == "pink":  # 1/f power: scale the spectrum by 1/sqrt(f)
            spectrum = np.fft.rfft(white)
            f = np.fft.rfftfreq(len(t), 1 / rate)
            spectrum[1:] /= np.sqrt(f[1:])
            spectrum[0] = 0
            white = np.fft.irfft(spectrum, len(t))
        wave = white / (np.abs(white).max() or 1.0)
    elif kind == "beeps":
        wave = np.sin(2 * np.pi * frequency * t) * ((t % 0.5) < 0.15)
    else:
        raise ValueError(f"unknown signal {kind!r} (known: {', '.join(GENERATORS)})")
    return amplitude * wave


def stft(signal: np.ndarray, rate: int, *, window_ms: float = 25.0, hop_ms: float = 10.0,
         n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(complex spectrum frames × bins, frame times in s, bin frequencies in Hz); Hann window."""
    window = max(int(rate * window_ms / 1000), 8)
    hop = max(int(rate * hop_ms / 1000), 1)
    n_fft = n_fft or 1 << (window - 1).bit_length()
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < window:
        signal = np.pad(signal, (0, window - len(signal)))
    count = 1 + (len(signal) - window) // hop
    frames = np.lib.stride_tricks.sliding_window_view(signal, window)[::hop][:count]
    spectrum = np.fft.rfft(frames * np.hanning(window), n=n_fft, axis=1)
    times = (np.arange(count) * hop + window / 2) / rate
    return spectrum, times, np.fft.rfftfreq(n_fft, 1 / rate)


def spectrogram(signal: np.ndarray, rate: int, **options: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(power in dB, frames × bins; times; frequencies)."""
    spectrum, times, freqs = stft(signal, rate, **options)
    power = np.abs(spectrum) ** 2
    return 10 * np.log10(power + 1e-10), times, freqs


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(bands: int, n_fft: int, rate: int, low: float = 0.0,
                   high: float | None = None) -> np.ndarray:
    """``bands × (n_fft // 2 + 1)`` triangular filters, each with unit area."""
    high = high or rate / 2
    edges = mel_to_hz(np.linspace(hz_to_mel(low), hz_to_mel(high), bands + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    bank = np.zeros((bands, len(freqs)))
    for b in range(bands):
        left, centre, right = edges[b], edges[b + 1], edges[b + 2]
        rise = (freqs - left) / max(centre - left, 1e-9)
        fall = (right - freqs) / max(right - centre, 1e-9)
        bank[b] = np.clip(np.minimum(rise, fall), 0, None)
        area = bank[b].sum()
        if area > 0:
            bank[b] /= area
    return bank


def mel_spectrogram(signal: np.ndarray, rate: int, bands: int = 64,
                    **options: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(log mel energies in dB, frames × bands; times; band centre frequencies)."""
    spectrum, times, freqs = stft(signal, rate, **options)
    n_fft = 2 * (spectrum.shape[1] - 1)
    bank = mel_filterbank(bands, n_fft, rate)
    energy = (np.abs(spectrum) ** 2) @ bank.T
    centres = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(rate / 2), bands + 2))[1:-1]
    return 10 * np.log10(energy + 1e-10), times, centres


def mfcc(signal: np.ndarray, rate: int, coefficients: int = 13, bands: int = 40,
         **options: float) -> np.ndarray:
    """``frames × coefficients`` MFCCs."""
    from scipy.fft import dct

    log_mel, _times, _centres = mel_spectrogram(signal, rate, bands, **options)
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :coefficients]


def bandpass(signal: np.ndarray, rate: int, low: float | None, high: float | None,
             order: int = 4) -> np.ndarray:
    """Butterworth filter run forwards and backwards (no phase shift): low-pass (``low`` None),
    high-pass (``high`` None) or band-pass."""
    from scipy.signal import butter, sosfiltfilt

    nyquist = rate / 2
    if low and high:
        sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    elif high:
        sos = butter(order, high / nyquist, btype="low", output="sos")
    elif low:
        sos = butter(order, low / nyquist, btype="high", output="sos")
    else:
        return np.asarray(signal, dtype=np.float64)
    return sosfiltfilt(sos, signal) if len(signal) > 3 * (2 * order + 1) else np.asarray(signal, dtype=float)


def envelope(signal: np.ndarray, rate: int, window_ms: float = 20.0) -> np.ndarray:
    """RMS loudness per window."""
    size = max(int(rate * window_ms / 1000), 1)
    count = len(signal) // size
    if not count:
        return np.zeros(0)
    return np.sqrt(np.mean(np.asarray(signal[:count * size]).reshape(count, size) ** 2, axis=1))


def pitch(signal: np.ndarray, rate: int, low: float = 60.0, high: float = 1000.0) -> float | None:
    """The fundamental frequency of a voiced or tonal signal by autocorrelation; ``None`` if none."""
    x = np.asarray(signal, dtype=np.float64)[: rate]
    x = x - x.mean()
    if len(x) < rate // high * 2 or not np.any(x):
        return None
    corr = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * len(x))) ** 2)[: len(x)]
    lo, hi = int(rate / high), min(int(rate / low), len(corr) - 1)
    if hi <= lo:
        return None
    lag = lo + int(np.argmax(corr[lo:hi]))
    return rate / lag if corr[lag] > 0.3 * corr[0] else None


def log_mel_image(signal: np.ndarray, rate: int, *, size: int = 64, seconds: float = 1.0) -> np.ndarray:
    """A clip as a ``size × size`` greyscale image (uint8): the log mel spectrogram of the first
    ``seconds`` (padded with silence), bands upwards, time left to right, 80 dB of range. This is
    how spoken digits become images that the vision networks train on."""
    target = 8_000
    x = resample(signal, rate, target)
    length = int(seconds * target)
    x = np.pad(x[:length], (0, max(0, length - len(x))))
    hop_ms = 1000 * seconds / size
    log_mel, _t, _c = mel_spectrogram(x, target, bands=size, window_ms=max(2 * hop_ms, 16.0), hop_ms=hop_ms)
    log_mel = log_mel[:size]
    if log_mel.shape[0] < size:
        log_mel = np.pad(log_mel, ((0, size - log_mel.shape[0]), (0, 0)), constant_values=log_mel.min())
    top = log_mel.max()
    scaled = np.clip((log_mel - (top - 80.0)) / 80.0, 0, 1)
    return (scaled.T[::-1] * 255).astype(np.uint8)
