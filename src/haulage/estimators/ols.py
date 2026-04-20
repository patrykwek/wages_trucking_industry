from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from haulage.assumptions import AssumptionSet
from haulage.estimands import ATE, Estimand
from haulage.estimators.base import BaseEstimator, EstimatorResult
from haulage.inference.cluster import cluster_one_way, cluster_two_way
from haulage.inference.driscoll_kraay import driscoll_kraay
from haulage.inference.hac import newey_west
from haulage.inference.sandwich import hc0, hc1, hc2, hc3
from haulage.utils.linalg import solve_ols
from haulage.utils.validation import check_shapes

SEKind = Literal[
    "nonrobust",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
    "cluster",
    "twoway",
    "hac",
    "driscoll_kraay",
]


@dataclass(frozen=True, slots=True)
class OLSResult(EstimatorResult):
    """OLS result, subclass of EstimatorResult carrying the coefficient vector and V."""

    coef: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cov: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    names: tuple[str, ...] = ()

    def coef_se(self, idx: int) -> float:
        return float(np.sqrt(self.cov[idx, idx]))

    def t(self, idx: int) -> float:
        return float(self.coef[idx] / self.coef_se(idx))


@dataclass
class OLS(BaseEstimator):
    r"""OLS with a battery of SEs.

    Implements $\hat\beta = (X'X)^{-1}X'y$ via rank-revealing QR. The `fit` method
    returns a result object whose `se` field reflects the chosen variance estimator.

    Args:
        se: One of "nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "cluster" | "twoway"
            | "hac" | "driscoll_kraay".
        dof_correction: Whether to apply small-sample adjustments where canonical.
        hac_lags: HAC truncation; None = Andrews (1991) optimal bandwidth.
        kernel: HAC kernel: "bartlett" | "parzen" | "qs".
        coef_of_interest: Index of the coefficient reported as `point`/`se`. Default 1
            (slot after constant); use 0 for intercept.
        names: Optional column names for reporting.

    References:
        White (1980); MacKinnon & White (1985); Liang & Zeger (1986); Cameron, Gelbach &
        Miller (2011); Newey & West (1987); Driscoll & Kraay (1998); Andrews (1991).
    """

    se: SEKind = "hc1"
    dof_correction: bool = True
    hac_lags: int | None = None
    kernel: str = "bartlett"
    coef_of_interest: int = 1
    name: str = "OLS"
    _names: tuple[str, ...] = ()

    @property
    def assumptions(self) -> AssumptionSet:
        from haulage.assumptions import Assumption

        return AssumptionSet(
            assumptions=(
                Assumption(
                    name="strict_exogeneity",
                    latex=r"\mathbb{E}[\epsilon \mid X] = 0",
                    description="Regressors strictly exogenous conditional on covariates.",
                    testable=False,
                ),
                Assumption(
                    name="no_multicollinearity",
                    latex=r"\mathrm{rank}(X) = k",
                    description="Full column rank; enforced numerically via rank-revealing QR.",
                    testable=True,
                ),
            ),
            estimator="OLS",
        )

    def fit(
        self,
        y: NDArray[np.float64],
        X: NDArray[np.float64] | None = None,
        *,
        clusters: NDArray[np.int64] | None = None,
        cluster_b: NDArray[np.int64] | None = None,
        time: NDArray[np.int64] | None = None,
        estimand: Estimand | None = None,
        names: list[str] | None = None,
        **_: Any,
    ) -> OLSResult:
        r"""Fit by OLS and return a canonical OLSResult.

        Args:
            y: Outcome (n,).
            X: Design (n, k) including any constant.
            clusters: One-way cluster labels, required when `se == "cluster"` or "twoway".
            cluster_b: Second cluster for two-way SE.
            time: Time index for HAC / Driscoll-Kraay.
            estimand: Estimand object stored with the result (default ATE()).
            names: Column names for the result object.

        Returns:
            OLSResult.
        """
        if X is None:
            raise ValueError("OLS.fit requires the design matrix X")
        check_shapes(y=y, X=X)
        n, k = X.shape
        beta, _R, rank = solve_ols(X, y)
        u = y - X @ beta
        V = self._covariance(X, u, clusters, cluster_b, time)
        i = self.coef_of_interest
        point = float(beta[i])
        se = float(np.sqrt(V[i, i]))
        col_names = tuple(names) if names is not None else tuple(f"x{j}" for j in range(k))
        diagnostics: dict[str, Any] = {
            "rank": rank,
            "sigma2": float(np.sum(u**2) / max(n - k, 1)),
            "r2": float(1 - np.var(u) / np.var(y)) if np.var(y) > 0 else float("nan"),
            "se_kind": self.se,
        }
        extra: dict[str, Any] = {"residuals": u}
        return OLSResult(
            estimand=estimand or ATE(),
            point=point,
            se=se,
            n=n,
            assumptions=self.assumptions,
            diagnostics=diagnostics,
            extra=extra,
            name=f"OLS[{self.se}]",
            coef=beta,
            cov=V,
            names=col_names,
        )

    def _covariance(
        self,
        X: NDArray[np.float64],
        u: NDArray[np.float64],
        clusters: NDArray[np.int64] | None,
        cluster_b: NDArray[np.int64] | None,
        time: NDArray[np.int64] | None,
    ) -> NDArray[np.float64]:
        n, k = X.shape
        if self.se == "nonrobust":
            sigma2 = float(np.sum(u**2) / max(n - k, 1))
            from haulage.utils.linalg import xtx_inv

            return sigma2 * xtx_inv(X)
        if self.se == "hc0":
            return hc0(X, u)
        if self.se == "hc1":
            return hc1(X, u)
        if self.se == "hc2":
            return hc2(X, u)
        if self.se == "hc3":
            return hc3(X, u)
        if self.se == "cluster":
            if clusters is None:
                raise ValueError("se='cluster' requires `clusters`")
            return cluster_one_way(X, u, clusters, dof_correction=self.dof_correction)
        if self.se == "twoway":
            if clusters is None or cluster_b is None:
                raise ValueError("se='twoway' requires both `clusters` and `cluster_b`")
            return cluster_two_way(X, u, clusters, cluster_b, dof_correction=self.dof_correction)
        if self.se == "hac":
            return newey_west(X, u, lags=self.hac_lags, kernel=self.kernel)
        if self.se == "driscoll_kraay":
            if time is None:
                raise ValueError("se='driscoll_kraay' requires `time`")
            return driscoll_kraay(X, u, time, lags=self.hac_lags, kernel=self.kernel)
        raise ValueError(f"unknown se {self.se!r}")
