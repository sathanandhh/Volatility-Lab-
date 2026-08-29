"""Innovation distributions supported by the model layer."""
from __future__ import annotations

import math

from scipy import stats


def var_quantile(dist: str, alpha: float, **params) -> float:
    """Inverse CDF for a standardized innovation."""
    if dist == "normal":
        return stats.norm.ppf(alpha)
    if dist == "t":
        nu = params.get("nu", 8.0)
        return stats.t.ppf(alpha, nu) * math.sqrt((nu - 2) / nu)
    if dist == "ged":
        nu = params.get("nu", 1.4)
        return stats.gennorm.ppf(alpha, beta=nu)
    if dist == "skewt":
        # Skew-Student-t — arch uses skewt with skew parameter
        # Use scipy.stats.nct as approximation
        nu = params.get("nu", 8.0)
        lam = params.get("lambda", 0.0)
        return stats.nct.ppf(alpha, df=nu, nc=lam)
    if dist == "jsu":
        a = params.get("a", 0.5)
        b = params.get("b", 1.0)
        return stats.johnsonsu.ppf(alpha, a=a, b=b)
    raise ValueError(f"Unknown distribution: {dist!r}")


def es_quantile(dist: str, alpha: float, **params) -> float:
    """Expected loss beyond the α-quantile (negative of mean tail)."""
    if dist == "normal":
        q = stats.norm.ppf(alpha)
        return stats.norm.pdf(q) / alpha
    if dist == "t":
        nu = params.get("nu", 8.0)
        q = stats.t.ppf(alpha, nu)
        scale = math.sqrt((nu - 2) / nu)
        # ES for Student-t: (f(q) * (nu + q²)) / ((nu - 1) * α) * scale
        return (stats.t.pdf(q, nu) * (nu + q ** 2)) / ((nu - 1) * alpha) * scale
    # Generic fallback via simulation
    samples = sample(dist, 100_000, **params)
    q = var_quantile(dist, alpha, **params)
    return float(samples[samples <= q].mean())


def sample(dist: str, n: int, **params):
    if dist == "normal":
        return stats.norm.rvs(size=n)
    if dist == "t":
        nu = params.get("nu", 8.0)
        scale = math.sqrt((nu - 2) / nu)
        return stats.t.rvs(df=nu, size=n) * scale
    if dist == "ged":
        nu = params.get("nu", 1.4)
        return stats.gennorm.rvs(beta=nu, size=n)
    if dist == "skewt":
        nu = params.get("nu", 8.0)
        lam = params.get("lambda", 0.0)
        return stats.nct.rvs(df=nu, nc=lam, size=n)
    if dist == "jsu":
        a = params.get("a", 0.5)
        b = params.get("b", 1.0)
        return stats.johnsonsu.rvs(a=a, b=b, size=n)
    raise ValueError(f"Unknown distribution: {dist!r}")
