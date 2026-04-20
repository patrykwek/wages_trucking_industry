from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Estimand:
    r"""Abstract causal target. Subclasses pin down the potential-outcomes functional.

    Attributes:
        name: Short tag used in tables and LaTeX output.
        latex: LaTeX expression of the functional.
        population: Human-readable description of the conditioning set.
    """

    name: str
    latex: str
    population: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class ATE(Estimand):
    r"""Average treatment effect: $\tau = \mathbb{E}[Y(1) - Y(0)]$."""

    name: str = "ATE"
    latex: str = r"\mathbb{E}[Y(1) - Y(0)]"
    population: str = "the full population"


@dataclass(frozen=True, slots=True)
class ATT(Estimand):
    r"""Average treatment effect on the treated: $\tau = \mathbb{E}[Y(1)-Y(0) \mid D=1]$."""

    name: str = "ATT"
    latex: str = r"\mathbb{E}[Y(1) - Y(0) \mid D=1]"
    population: str = "the treated units"


@dataclass(frozen=True, slots=True)
class ATTgt(Estimand):
    r"""Callaway-Sant'Anna group-time ATT $ATT(g,t) = \mathbb{E}[Y_t(g) - Y_t(\infty) \mid G=g]$."""

    g: int = 0
    t: int = 0
    name: str = "ATT(g,t)"
    latex: str = r"\mathbb{E}[Y_t(g) - Y_t(\infty) \mid G=g]"
    population: str = "units first treated at time g"

    def __str__(self) -> str:
        return f"ATT(g={self.g}, t={self.t})"


@dataclass(frozen=True, slots=True)
class EventStudyATT(Estimand):
    r"""Event-study aggregation $ATT^{es}(e) = \sum_g w_g\, ATT(g, g+e)$."""

    event_time: int = 0
    name: str = "ATT^es"
    latex: str = r"\sum_g w_g\, ATT(g, g+e)"
    population: str = "cohorts at event time e"

    def __str__(self) -> str:
        return f"ATT^es(e={self.event_time})"


@dataclass(frozen=True, slots=True)
class LATE(Estimand):
    r"""Local average treatment effect among compliers (Imbens-Angrist 1994)."""

    name: str = "LATE"
    latex: str = r"\mathbb{E}[Y(1) - Y(0) \mid \text{complier}]"
    population: str = "compliers"


@dataclass(frozen=True, slots=True)
class CATE(Estimand):
    r"""Conditional average treatment effect $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]$."""

    covariates: tuple[str, ...] = field(default_factory=tuple)
    name: str = "CATE"
    latex: str = r"\mathbb{E}[Y(1) - Y(0) \mid X=x]"
    population: str = "units with covariates x"


@dataclass(frozen=True, slots=True)
class QTE(Estimand):
    r"""Quantile treatment effect $\tau(\tau_q) = F_{Y(1)}^{-1}(\tau_q) - F_{Y(0)}^{-1}(\tau_q)$."""

    quantile: float = 0.5
    name: str = "QTE"
    latex: str = r"F_{Y(1)}^{-1}(\tau) - F_{Y(0)}^{-1}(\tau)"
    population: str = "the full population, at quantile tau"

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile < 1.0:
            raise ValueError(f"quantile must be in (0,1), got {self.quantile}")


@dataclass(frozen=True, slots=True)
class SharpRD(Estimand):
    r"""Sharp RD effect at cutoff $c$:
    $\tau_{SRD} = \lim_{x\downarrow c}\mathbb{E}[Y|X{=}x] - \lim_{x\uparrow c}\mathbb{E}[Y|X{=}x]$.
    """

    cutoff: float = 0.0
    running: str = "x"
    name: str = "SRD"
    latex: str = r"\lim_{x\downarrow c}\mathbb{E}[Y|X=x] - \lim_{x\uparrow c}\mathbb{E}[Y|X=x]"
    population: str = "units at the cutoff"
