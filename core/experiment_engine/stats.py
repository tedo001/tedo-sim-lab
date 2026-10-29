"""Statistics for repeated measurements: several seeds of one experiment, or repeated timing
blocks of a benchmark.

``summarize`` gives n, mean, sample standard deviation, standard error and a two-sided 95 %
confidence interval for the mean from Student's t distribution (n − 1 degrees of freedom), which
is the right interval for a handful of seeds; with one value there is no spread and no interval.
``welch`` compares two sets without assuming equal variances (Welch's t-test, Welch–Satterthwaite
degrees of freedom) and gives the difference of the means with its 95 % interval. Every result
says how it was computed (``method``), so an exported number can be traced to its formula."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

__all__ = ["CONFIDENCE", "summarize", "t_quantile", "welch"]

CONFIDENCE = 0.95


def t_quantile(df: float, confidence: float = CONFIDENCE) -> float:
    """The two-sided Student-t critical value (scipy when present, else a close series)."""
    try:
        from scipy import stats

        return float(stats.t.ppf(0.5 + confidence / 2, df))
    except ImportError:  # invert the exact tail probability by bisection
        target, low, high = 1 - confidence, 0.0, 1e3
        for _ in range(200):
            middle = (low + high) / 2
            if _t_two_sided(middle, df) > target:
                low = middle
            else:
                high = middle
        return (low + high) / 2


def _t_two_sided(t: float, df: float) -> float:
    return _betainc(df / 2, 0.5, df / (df + t * t))


def _t_sf(t: float, df: float) -> float:
    """Two-sided p-value of ``t`` with ``df`` degrees of freedom."""
    try:
        from scipy import stats

        return float(2 * stats.t.sf(abs(t), df))
    except ImportError:
        return _t_two_sided(t, df)


def _betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function (continued fraction, Numerical Recipes)."""
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x)
                     + b * math.log(1 - x))
    if x > (a + 1) / (a + b + 2):
        return 1 - _betainc(b, a, 1 - x)
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > 1e-30 else 1e-30)
    f = d
    for m in range(1, 200):
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                          -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + numerator * d
            d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + numerator / c
            c = c if abs(c) > 1e-30 else 1e-30
            f *= c * d
    return front * f / a


def summarize(values: Sequence[float], confidence: float = CONFIDENCE) -> dict[str, Any]:
    """n, mean, sd, sem, min, max and the t-interval of the mean."""
    data = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(data)
    if n == 0:
        return {"n": 0, "method": "no values"}
    mean = sum(data) / n
    out: dict[str, Any] = {"n": n, "mean": mean, "min": min(data), "max": max(data)}
    if n == 1:
        return {**out, "sd": None, "sem": None, "ci_low": None, "ci_high": None,
                "method": "one value: no spread and no interval (run more seeds)"}
    sd = math.sqrt(sum((v - mean) ** 2 for v in data) / (n - 1))
    sem = sd / math.sqrt(n)
    half = t_quantile(n - 1, confidence) * sem
    return {**out, "sd": sd, "sem": sem, "ci_low": mean - half, "ci_high": mean + half,
            "confidence": confidence,
            "method": f"mean ± t(0.975, df={n - 1}) × sd/√n; sd with n − 1 (sample standard deviation)"}


def welch(first: Sequence[float], second: Sequence[float], confidence: float = CONFIDENCE) -> dict[str, Any]:
    """Welch's t-test of ``second`` against ``first``: difference (second − first), its interval,
    t, degrees of freedom and the two-sided p-value."""
    a, b = summarize(first, confidence), summarize(second, confidence)
    if a["n"] < 2 or b["n"] < 2:
        return {"difference": (b.get("mean") or 0) - (a.get("mean") or 0) if a["n"] and b["n"] else None,
                "p_value": None, "method": "Welch's t-test needs at least two values on each side"}
    va, vb = a["sd"] ** 2 / a["n"], b["sd"] ** 2 / b["n"]
    difference = b["mean"] - a["mean"]
    se = math.sqrt(va + vb)
    if se == 0:
        return {"difference": difference, "t": None, "df": None, "p_value": 0.0 if difference else 1.0,
                "ci_low": difference, "ci_high": difference, "method": "no spread on either side"}
    df = (va + vb) ** 2 / (va ** 2 / (a["n"] - 1) + vb ** 2 / (b["n"] - 1))
    t = difference / se
    half = t_quantile(df, confidence) * se
    return {"difference": difference, "t": t, "df": df, "p_value": _t_sf(t, df),
            "ci_low": difference - half, "ci_high": difference + half, "confidence": confidence,
            "method": "Welch's t-test (unequal variances, Welch–Satterthwaite df), two-sided"}
