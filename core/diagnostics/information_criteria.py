"""Information criteria: AIC, BIC, HQIC — extracted from arch fit or computed directly."""
from __future__ import annotations


def aic(fit) -> float:
    return float(fit.aic)


def bic(fit) -> float:
    return float(fit.bic)


def hqic(fit) -> float:
    """Hannan-Quinn — arch fit may not expose it; compute manually."""
    try:
        return float(fit.hqic)
    except AttributeError:
        import numpy as np
        k = int(getattr(fit, "n_params", 1))
        n = len(fit.resid)
        ll = float(fit.loglikelihood)
        return float(-2 * ll + 2 * k * np.log(np.log(max(n, 3))))
