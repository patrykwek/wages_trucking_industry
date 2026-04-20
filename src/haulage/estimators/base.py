from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from haulage.assumptions import AssumptionSet
from haulage.estimands import Estimand
from haulage.influence import InfluenceFunction


@dataclass(frozen=True, slots=True)
class EstimatorResult:
    r"""Canonical output object for every estimator in the package.

    Attributes:
        estimand: The causal target.
        point: Point estimate $\hat\tau$.
        se: Default standard error (estimator-specific).
        n: Effective sample size used.
        assumptions: Identifying assumption set.
        diagnostics: Estimator-specific diagnostic dict.
        influence_: Influence-function object, or None.
        extra: Free-form slot for estimator-specific attributes (coefficients, fits, ...).
        name: Short tag for tables and plots.
    """

    estimand: Estimand
    point: float
    se: float
    n: int
    assumptions: AssumptionSet
    diagnostics: dict[str, Any] = field(default_factory=dict)
    influence_: InfluenceFunction | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    name: str = "estimator"

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Wald two-sided confidence interval at level 1 - alpha."""
        from scipy.stats import norm

        z = float(norm.ppf(1 - alpha / 2))
        return (self.point - z * self.se, self.point + z * self.se)

    @property
    def z(self) -> float:
        """Wald z-statistic for H0: tau = 0."""
        return float(self.point / self.se) if self.se > 0 else float("nan")

    @property
    def pvalue(self) -> float:
        """Two-sided asymptotic p-value from a normal approximation."""
        from scipy.stats import norm

        if not np.isfinite(self.z):
            return float("nan")
        return float(2 * (1 - norm.cdf(abs(self.z))))

    def to_markdown(self) -> str:
        """One-line markdown summary."""
        lo, hi = self.ci()
        return (
            f"**{self.name}** ({self.estimand}): "
            f"$\\hat\\tau = {self.point:.4f}$ (SE {self.se:.4f}), "
            f"95% CI [{lo:.4f}, {hi:.4f}], n = {self.n}"
        )

    def to_latex(self) -> str:
        """LaTeX table row: tau, (SE), [CI]."""
        lo, hi = self.ci()
        return (
            f"{self.name} & ${self.point:.4f}$ & $({self.se:.4f})$ & $[{lo:.4f}, {hi:.4f}]$ & ${self.n}$ \\\\"
        )


class BaseEstimator(ABC):
    """Abstract base for every haulage estimator.

    Subclasses implement `fit` returning an `EstimatorResult`. The constructor stores
    configuration; fit accepts the data. This split lets us re-use the same estimator
    object in pipelines (sensitivity, cross-fitting) without re-instantiation.
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        y: NDArray[np.float64] | Any,
        *args: Any,
        **kwargs: Any,
    ) -> EstimatorResult:
        """Fit the estimator and return the canonical result object."""
        raise NotImplementedError

    @property
    @abstractmethod
    def assumptions(self) -> AssumptionSet:
        """Identifying assumptions for this estimator."""
        raise NotImplementedError
