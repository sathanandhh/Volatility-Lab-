"""Model registry: catalog of supported models and their specifications."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelSpec:
    """Specification of a supported volatility model."""
    name: str
    family: str  # "univariate", "multivariate", "stochastic", "ml"
    vol: str = "GARCH"  # arch-family keyword
    default_p: int = 1
    default_q: int = 1
    default_o: int = 0
    supports_asymmetry: bool = False
    supported_distributions: list[str] = field(default_factory=lambda: ["normal", "t", "ged", "skewt", "jsu"])
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "vol": self.vol,
            "default_p": self.default_p,
            "default_q": self.default_q,
            "default_o": self.default_o,
            "supports_asymmetry": self.supports_asymmetry,
            "supported_distributions": self.supported_distributions,
            "description": self.description,
        }


_REGISTRY: dict[str, ModelSpec] = {
    "ARCH": ModelSpec(
        name="ARCH", family="univariate", vol="ARCH",
        default_p=1, default_q=0, default_o=0,
        supports_asymmetry=False,
        description="Engle 1982. Variance reacts to lagged squared shocks only.",
    ),
    "GARCH": ModelSpec(
        name="GARCH", family="univariate", vol="GARCH",
        default_p=1, default_q=1, default_o=0,
        supports_asymmetry=False,
        description="Bollerslev 1986. Adds lagged variance — persistence.",
    ),
    "GJR-GARCH": ModelSpec(
        name="GJR-GARCH", family="univariate", vol="GARCH",
        default_p=1, default_q=1, default_o=1,
        supports_asymmetry=True,
        description="Glosten-Jagannathan-Runkle 1993. Threshold asymmetry.",
    ),
    "EGARCH": ModelSpec(
        name="EGARCH", family="univariate", vol="EGARCH",
        default_p=1, default_q=1, default_o=1,
        supports_asymmetry=True,
        description="Nelson 1991. Log-variance; positivity guaranteed; smooth leverage.",
    ),
    "FIGARCH": ModelSpec(
        name="FIGARCH", family="univariate", vol="FIGARCH",
        default_p=1, default_q=1, default_o=0,
        supports_asymmetry=False,
        description="Fractionally integrated GARCH; long memory in volatility.",
    ),
    "IGARCH": ModelSpec(
        name="IGARCH", family="univariate", vol="IGARCH",
        default_p=1, default_q=1, default_o=0,
        supports_asymmetry=False,
        description="Integrated GARCH; persistence = 1; non-stationary.",
    ),
    "EWMA": ModelSpec(
        name="EWMA", family="univariate", vol="EWMA",
        default_p=0, default_q=0, default_o=0,
        supported_distributions=["normal"],
        description="RiskMetrics 1996. Exponentially weighted moving average.",
    ),
    "HARCH": ModelSpec(
        name="HARCH", family="univariate", vol="HARCH",
        default_p=3, default_q=0, default_o=0,
        description="Heterogeneous ARCH — captures long memory via multi-scale lags.",
    ),
    "DCC-GARCH": ModelSpec(
        name="DCC-GARCH", family="multivariate", vol="DCC",
        description="Engle 2002. Time-varying cross-asset correlations.",
    ),
    "BEKK": ModelSpec(
        name="BEKK", family="multivariate", vol="BEKK",
        description="Engle-Kroner 1995. Positive-definite covariance without constraints.",
    ),
    "GO-GARCH": ModelSpec(
        name="GO-GARCH", family="multivariate", vol="GO-GARCH",
        description="van der Weide 2002. Orthogonal factors drive covariance.",
    ),
    "Realized-GARCH": ModelSpec(
        name="Realized-GARCH", family="univariate", vol="Realized-GARCH",
        description="Hansen-Huang-Shek 2012. Uses intraday realized variance.",
    ),
    "HEAVY": ModelSpec(
        name="HEAVY", family="univariate", vol="HEAVY",
        description="Shephard-Andersen 2009. High-frEquency-based volatility.",
    ),
    "Heston": ModelSpec(
        name="Heston", family="stochastic", vol="Heston",
        description="Affine stochastic volatility; closed-form option pricing.",
    ),
    "Heston-Nandi": ModelSpec(
        name="Heston-Nandi", family="stochastic", vol="Heston-Nandi",
        description="GARCH-like SV with closed-form option pricing.",
    ),
    "SV-Jumps": ModelSpec(
        name="SV-Jumps", family="stochastic", vol="SV-Jumps",
        description="Stochastic volatility with Poisson jump component.",
    ),
    "LSTM-Vol": ModelSpec(
        name="LSTM-Vol", family="ml", vol="LSTM",
        description="Sequence model on squared returns.",
    ),
    "Transformer-Vol": ModelSpec(
        name="Transformer-Vol", family="ml", vol="Transformer",
        description="Attention-based; long-context volatility.",
    ),
    "TFT": ModelSpec(
        name="TFT", family="ml", vol="TFT",
        description="Temporal Fusion Transformer; multi-horizon forecasting.",
    ),
}


def list_models(family: str | None = None) -> list[ModelSpec]:
    if family is None:
        return list(_REGISTRY.values())
    return [s for s in _REGISTRY.values() if s.family == family]


def get_model_spec(name: str) -> ModelSpec | None:
    return _REGISTRY.get(name)
