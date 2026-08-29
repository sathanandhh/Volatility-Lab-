"""Sign-bias tests (Engle-Ng 1993) on standardized residuals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sign_bias_test(std_resid: pd.Series) -> dict:
    """Negative and positive sign-bias tests."""
    from scipy import stats
    s = std_resid.dropna()
    if len(s) < 30:
        return {"negative_t": float("nan"), "negative_p": float("nan"),
                "positive_t": float("nan"), "positive_p": float("nan")}
    sq = s.pow(2).to_numpy()
    neg = (s < 0).astype(float).to_numpy()
    pos = (s >= 0).astype(float).to_numpy()
    # Negative sign-bias regression
    t_neg, p_neg = _regress_t(sq, neg)
    t_pos, p_pos = _regress_t(sq, pos)
    return {
        "negative_t": float(t_neg), "negative_p": float(p_neg),
        "positive_t": float(t_pos), "positive_p": float(p_pos),
    }


def _regress_t(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    n = len(y)
    X = np.column_stack([np.ones_like(x), x])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = resid @ resid / (n - 2)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t = beta[1] / se[1] if se[1] > 0 else 0.0
        from scipy import stats as ss
        p = 2 * (1 - ss.t.cdf(abs(t), df=n - 2))
        return t, p
    except Exception:
        return float("nan"), float("nan")
