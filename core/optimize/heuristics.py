"""Rules-of-thumb docstring bank for input selection.

This module has no executable code — it exists to be read by the LLM
advisor when explaining why an input was chosen.
"""

HEURISTICS = {
    "aic_vs_bic": (
        "AIC is preferred for prediction (less penalty for complexity); "
        "BIC is preferred when the goal is true-model identification "
        "(heavier penalty for parameters)."
    ),
    "aic_delta": (
        "ΔAIC < 2 → both models have substantial support; prefer simpler. "
        "ΔAIC 4-7 → less support for the more complex. "
        "ΔAIC > 10 → essentially no support for the more complex."
    ),
    "qlike_preference": (
        "QLIKE is the preferred loss function for volatility-forecast "
        "evaluation (Patton 2011). It is robust to noise in the "
        "realized-variance proxy (squared returns)."
    ),
    "distribution_when_normality_blocked": (
        "If preflight normality is blocked, Normal innovations will "
        "almost always lose the AIC sweep. Student-t is the standard "
        "first choice; skew-t when asymmetry is also detected."
    ),
    "refit_tradeoff": (
        "Shorter refit intervals adapt faster to regime shifts but cost "
        "more compute. The optimal is found by maximizing conditional-"
        "coverage p-value from the backtest."
    ),
    "window_tradeoff": (
        "Short rolling windows react quickly but are noisy. Long windows "
        "are smooth but lag regime changes. Pick the one minimizing "
        "out-of-sample QLIKE."
    ),
    "egarch_multistep": (
        "EGARCH has no closed-form multi-step forecast beyond one period. "
        "Use simulation-based forecasting for horizon > 1."
    ),
    "arch_lm_critical": (
        "Engle ARCH-LM is the single most important pre-flight check. "
        "If it fails to reject the null of no ARCH effect, the entire "
        "GARCH family is statistically unjustified — use EWMA instead."
    ),
}


def get_heuristic(key: str) -> str:
    return HEURISTICS.get(key, f"No heuristic found for '{key}'.")
