from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge

from haulage.assumptions import (
    NO_ANTICIPATION,
    OVERLAP,
    PARALLEL_TRENDS,
    SUTVA,
    AssumptionSet,
)
from haulage.estimands import ATTgt, EventStudyATT
from haulage.estimators.base import BaseEstimator, EstimatorResult
from haulage.influence import InfluenceFunction
from haulage.utils.validation import ensure_panel

Estimator = Literal["dr", "ipw", "or"]


@dataclass(frozen=True, slots=True)
class ATTgtEstimate:
    """A single ATT(g, t) with its influence-function contributions.

    Attributes:
        g: First-treatment cohort.
        t: Calendar time.
        point: ATT(g, t) estimate.
        influence: (n_in_cell,) influence-function values for analytical inference.
        n_cell: Number of treated units used.
    """

    g: int
    t: int
    point: float
    influence: NDArray[np.float64]
    n_cell: int


@dataclass
class CallawaySantAnna(BaseEstimator):
    r"""Callaway-Sant'Anna (2021) group-time ATTs with DR, IPW, or OR estimation.

    Nuisances fit on never-treated (or not-yet-treated) comparisons with cross-fitting
    turned off by default for the base implementation; cross-fitting can be added in a
    wrapper. Aggregates to event-study, overall, and group-level ATTs.

    Args:
        treatment: Binary treatment column.
        outcome: Outcome column.
        unit: Unit id.
        time: Time id.
        covariates: Pre-treatment covariates used in the nuisance regressions.
        control: "nevertreated" (default) or "notyettreated".
        method: "dr" | "ipw" | "or".
        anticipation: Pre-treatment periods assumed anticipation-free (default 0).

    References:
        Callaway & Sant'Anna (2021), *Difference-in-differences with multiple time
        periods*, Journal of Econometrics, 225(2): 200-230.
    """

    treatment: str = "D"
    outcome: str = "y"
    unit: str = "unit"
    time: str = "t"
    covariates: tuple[str, ...] = ()
    control: Literal["nevertreated", "notyettreated"] = "nevertreated"
    method: Estimator = "dr"
    anticipation: int = 0
    name: str = "CS(2021)"
    _fitted_gt: list[ATTgtEstimate] = field(default_factory=list)

    @property
    def assumptions(self) -> AssumptionSet:
        return AssumptionSet(
            assumptions=(PARALLEL_TRENDS, NO_ANTICIPATION, OVERLAP, SUTVA),
            estimator="CS",
        )

    def fit(self, y: Any, *args: Any, **kwargs: Any) -> EstimatorResult:
        if not isinstance(y, pd.DataFrame):
            raise TypeError("CS.fit expects a pandas DataFrame")
        df = ensure_panel(y, self.unit, self.time)
        df = df.drop(columns=[c for c in ("g",) if c in df.columns])
        first = df.loc[df[self.treatment] == 1].groupby(self.unit)[self.time].min().rename("g")
        df = df.merge(first, how="left", left_on=self.unit, right_index=True)
        df["g"] = df["g"].fillna(np.inf)

        times = np.sort(df[self.time].unique())
        groups = sorted(set(df["g"].dropna().unique().tolist()) - {np.inf})
        estimates: list[ATTgtEstimate] = []
        for g in groups:
            g_int = int(g)
            for t in times:
                if t == g_int - 1 - self.anticipation:
                    continue
                est = self._att_gt(df, g_int, int(t), times)
                if est is not None:
                    estimates.append(est)
        self._fitted_gt = estimates

        overall_point, overall_se, overall_infl = self._aggregate_overall(estimates)
        infl = InfluenceFunction(values=overall_infl, tau_hat=overall_point, name="CS-overall-ATT")
        return EstimatorResult(
            estimand=ATTgt(g=0, t=0),
            point=overall_point,
            se=overall_se,
            n=int(overall_infl.shape[0]),
            assumptions=self.assumptions,
            diagnostics={
                "n_groups": len(groups),
                "n_gt_cells": len(estimates),
                "method": self.method,
                "control": self.control,
            },
            influence_=infl,
            extra={"gt": estimates},
            name=f"CS[{self.method}]",
        )

    def event_study(self) -> pd.DataFrame:
        """Aggregate fitted ATT(g, t) into an event-study by e = t - g."""
        if not self._fitted_gt:
            raise RuntimeError("call fit() before event_study()")
        rows: list[dict[str, Any]] = []
        for est in self._fitted_gt:
            e = est.t - est.g
            rows.append(
                {
                    "event_time": e,
                    "g": est.g,
                    "t": est.t,
                    "att": est.point,
                    "n_cell": est.n_cell,
                    "se_cell": float(np.std(est.influence, ddof=1) / np.sqrt(est.n_cell))
                    if est.n_cell > 1
                    else np.nan,
                }
            )
        df = pd.DataFrame(rows)
        agg = (
            df.groupby("event_time", group_keys=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "att": np.average(g["att"], weights=g["n_cell"]),
                        "n": int(g["n_cell"].sum()),
                        "n_cohorts": int(g.shape[0]),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        agg["estimand"] = agg["event_time"].apply(lambda e: str(EventStudyATT(event_time=int(e))))
        return agg

    def _att_gt(self, df: pd.DataFrame, g: int, t: int, times: NDArray[np.int64]) -> ATTgtEstimate | None:
        t_pre = g - 1 - self.anticipation
        if t_pre not in times or t not in times:
            return None
        ctrl_mask = np.isinf(df["g"]) if self.control == "nevertreated" else (df["g"] > t).fillna(True)
        treat_mask = df["g"] == g
        use = treat_mask | ctrl_mask
        sub_pre = df[use & (df[self.time] == t_pre)]
        sub_post = df[use & (df[self.time] == t)]
        if sub_pre.empty or sub_post.empty:
            return None
        merged = sub_pre.merge(sub_post, on=self.unit, suffixes=("_pre", "_post"))
        if merged.empty:
            return None
        dY = (merged[f"{self.outcome}_post"] - merged[f"{self.outcome}_pre"]).to_numpy(dtype=np.float64)
        D = (merged[f"{self.treatment}_post"] > 0).to_numpy().astype(np.float64)
        D[(merged["g_post"] == g).to_numpy()] = 1.0
        D[D != 1.0] = 0.0
        if np.all(D == 0) or np.all(D == 1):
            return None
        X_pre = (
            merged[[f"{c}_pre" for c in self.covariates]].to_numpy(dtype=np.float64)
            if self.covariates
            else np.zeros((merged.shape[0], 0))
        )
        point, infl = _dr_att(dY, D, X_pre, method=self.method)
        return ATTgtEstimate(g=g, t=t, point=point, influence=infl, n_cell=int(D.sum()))

    def _aggregate_overall(self, estimates: list[ATTgtEstimate]) -> tuple[float, float, NDArray[np.float64]]:
        if not estimates:
            return float("nan"), float("nan"), np.zeros(0)
        post = [e for e in estimates if e.t >= e.g]
        if not post:
            post = estimates
        weights = np.array([e.n_cell for e in post], dtype=np.float64)
        weights /= weights.sum()
        point = float(sum(w * e.point for w, e in zip(weights, post, strict=False)))
        var_cells = np.array(
            [float(np.var(e.influence, ddof=1) / max(e.influence.shape[0], 1)) for e in post]
        )
        se = float(np.sqrt(np.sum((weights**2) * var_cells)))
        infl = np.concatenate([e.influence for e in post])
        return point, se, infl


def _dr_att(
    dY: NDArray[np.float64], D: NDArray[np.float64], X: NDArray[np.float64], method: Estimator
) -> tuple[float, NDArray[np.float64]]:
    r"""DR/IPW/OR ATT on the delta-outcome with optional covariate adjustment.

    Implements the Sant'Anna & Zhao (2020) DR estimator
    $\widehat{ATT} = \mathbb{E}\!\left[\left(D/\hat p - (1-D)\hat e(X) / ((1-\hat e(X))\hat p)\right)
    (\Delta Y - \hat m^0(X))\right]$.
    """
    n = dY.shape[0]
    if method == "or" or (method == "dr" and X.shape[1] > 0):
        mu0 = _fit_mu0(dY, D, X)
    else:
        mu0 = np.full(n, float(np.mean(dY[D == 0])) if np.any(D == 0) else 0.0)
    if method == "ipw" or (method == "dr" and X.shape[1] > 0):
        pscore = _fit_pscore(D, X)
    else:
        pscore = np.full(n, float(np.mean(D)))
    pscore = np.clip(pscore, 1e-3, 1 - 1e-3)
    p_bar = float(np.mean(D))

    if method == "or":
        pred1 = dY[D == 1].mean() if np.any(D == 1) else 0.0
        point = float(pred1 - np.mean(mu0[D == 1])) if np.any(D == 1) else 0.0
        infl = np.where(D == 1, (dY - mu0) - point, 0.0) / max(p_bar, 1e-8)
        return point, infl
    if method == "ipw":
        w1 = D / p_bar
        w0 = (1 - D) * pscore / (1 - pscore) / p_bar
        point = float(np.mean((w1 - w0) * dY))
        infl = (w1 - w0) * dY - point
        return point, infl
    # DR
    m1_term = D / p_bar
    m0_term = (1 - D) * pscore / (1 - pscore) / p_bar
    res = dY - mu0
    point = float(np.mean((m1_term - m0_term) * res))
    infl = (m1_term - m0_term) * res - point
    return point, infl


def _fit_mu0(dY: NDArray[np.float64], D: NDArray[np.float64], X: NDArray[np.float64]) -> NDArray[np.float64]:
    mask = D == 0
    if mask.sum() < max(5, X.shape[1] + 1) or X.shape[1] == 0:
        return np.full(dY.shape[0], float(dY[mask].mean()) if mask.any() else 0.0)
    mdl = Ridge(alpha=1.0).fit(X[mask], dY[mask])
    out: NDArray[np.float64] = mdl.predict(X).astype(np.float64)
    return out


def _fit_pscore(D: NDArray[np.float64], X: NDArray[np.float64]) -> NDArray[np.float64]:
    if X.shape[1] == 0:
        return np.full(D.shape[0], float(D.mean()))
    mdl = LogisticRegression(max_iter=500).fit(X, D.astype(int))
    out: NDArray[np.float64] = mdl.predict_proba(X)[:, 1].astype(np.float64)
    return out
