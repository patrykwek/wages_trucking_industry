from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from haulage.utils.validation import ensure_panel


@dataclass(frozen=True, slots=True)
class BaconComponent:
    """One of the three TWFE comparison types in the Goodman-Bacon (2021) decomposition.

    Attributes:
        kind: "treated_vs_never" | "early_vs_later_control" | "later_vs_already_treated".
        g_treat: Treatment cohort for the treated leg.
        g_control: Treatment cohort used as control (np.inf for never-treated).
        weight: Bacon weight attached to this 2x2 comparison.
        estimate: 2x2 DiD estimate for this comparison.
    """

    kind: str
    g_treat: float
    g_control: float
    weight: float
    estimate: float


@dataclass(frozen=True, slots=True)
class BaconDecomposition:
    """Weighted decomposition of the TWFE estimator into 2x2 comparisons.

    Attributes:
        components: All identified 2x2 comparisons with weights and estimates.
        total: Sum of weight * estimate, equal to the TWFE coefficient on D.
        negative_weight_share: Fraction of Bacon weight that is negative.
    """

    components: tuple[BaconComponent, ...]
    total: float
    negative_weight_share: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "kind": c.kind,
                    "g_treat": c.g_treat,
                    "g_control": c.g_control,
                    "weight": c.weight,
                    "estimate": c.estimate,
                }
                for c in self.components
            ]
        )

    def aggregate_by_kind(self) -> pd.DataFrame:
        """Sum weight and (weighted) estimate by comparison kind."""
        df = self.to_frame()
        g = df.groupby("kind").agg(
            weight=("weight", "sum"),
            est=(
                "estimate",
                lambda s: float(
                    np.sum(s * df.loc[s.index, "weight"]) / max(df.loc[s.index, "weight"].sum(), 1e-12)
                ),
            ),
        )
        return g.reset_index()


@dataclass
class GoodmanBacon:
    r"""Goodman-Bacon (2021) decomposition of TWFE with a staggered binary treatment.

    Args:
        treatment: Binary treatment column.
        outcome: Outcome column.
        unit: Unit identifier.
        time: Time identifier.

    References:
        Goodman-Bacon (2021), *Difference-in-differences with variation in treatment
        timing*, Journal of Econometrics.
    """

    treatment: str = "D"
    outcome: str = "y"
    unit: str = "unit"
    time: str = "t"

    def fit(self, panel: pd.DataFrame) -> BaconDecomposition:
        df = ensure_panel(panel, self.unit, self.time)
        df = df.drop(columns=[c for c in ("g",) if c in df.columns])
        first_treat = df.loc[df[self.treatment] == 1].groupby(self.unit)[self.time].min().rename("g")
        df = df.merge(first_treat, how="left", left_on=self.unit, right_index=True)
        df["g"] = df["g"].fillna(np.inf)

        groups = sorted(df["g"].unique().tolist(), key=lambda x: (np.isinf(x), x))
        comps: list[BaconComponent] = []
        times = np.array(sorted(df[self.time].unique()))
        df.shape[0]

        for gt in groups:
            if np.isinf(gt):
                continue
            for gc in groups:
                if gc == gt:
                    continue
                if not np.isinf(gc) and gc <= gt:
                    continue
                comp = _two_by_two(df, gt, gc, times, self.outcome, self.unit, self.time)
                if comp is None:
                    continue
                kind = "treated_vs_never" if np.isinf(gc) else "early_vs_later_control"
                comps.append(
                    BaconComponent(kind=kind, g_treat=gt, g_control=gc, weight=comp[0], estimate=comp[1])
                )

        for gt in groups:
            if np.isinf(gt):
                continue
            for gc in groups:
                if np.isinf(gc) or gc >= gt:
                    continue
                comp = _two_by_two_later_vs_already(df, gt, gc, times, self.outcome, self.unit, self.time)
                if comp is None:
                    continue
                comps.append(
                    BaconComponent(
                        kind="later_vs_already_treated",
                        g_treat=gt,
                        g_control=gc,
                        weight=comp[0],
                        estimate=comp[1],
                    )
                )

        tot_w = sum(c.weight for c in comps)
        if tot_w > 0:
            comps_final: tuple[BaconComponent, ...] = tuple(
                BaconComponent(c.kind, c.g_treat, c.g_control, c.weight / tot_w, c.estimate) for c in comps
            )
        else:
            comps_final = tuple(comps)
        total = float(sum(c.weight * c.estimate for c in comps_final))
        neg = float(sum(c.weight for c in comps_final if c.weight < 0))
        return BaconDecomposition(components=comps_final, total=total, negative_weight_share=neg)


def _cohort_mean(df: pd.DataFrame, g: float, t0: int, t1: int, outcome: str, time: str) -> float:
    sub = df[(df["g"] == g) & (df[time] >= t0) & (df[time] <= t1)]
    return float(sub[outcome].mean()) if sub.shape[0] else float("nan")


def _two_by_two(
    df: pd.DataFrame, gt: float, gc: float, times: NDArray[np.int64], outcome: str, unit: str, time: str
) -> tuple[float, float] | None:
    if np.isinf(gc):
        pre = times[times < gt]
        post = times[times >= gt]
    else:
        pre = times[times < gt]
        post = times[(times >= gt) & (times < gc)]
    if pre.size == 0 or post.size == 0:
        return None
    y_t_post = _cohort_mean(df, gt, post.min(), post.max(), outcome, time)
    y_t_pre = _cohort_mean(df, gt, pre.min(), pre.max(), outcome, time)
    y_c_post = _cohort_mean(df, gc, post.min(), post.max(), outcome, time)
    y_c_pre = _cohort_mean(df, gc, pre.min(), pre.max(), outcome, time)
    if any(np.isnan([y_t_post, y_t_pre, y_c_post, y_c_pre])):
        return None
    est = (y_t_post - y_t_pre) - (y_c_post - y_c_pre)
    n_t = int((df["g"] == gt).sum())
    n_c = int((df["g"] == gc).sum()) if not np.isinf(gc) else int((df["g"] == np.inf).sum())
    w = (n_t + n_c) * (pre.size * post.size)
    return float(w), float(est)


def _two_by_two_later_vs_already(
    df: pd.DataFrame, gt: float, gc: float, times: NDArray[np.int64], outcome: str, unit: str, time: str
) -> tuple[float, float] | None:
    pre = times[(times >= gc) & (times < gt)]
    post = times[times >= gt]
    if pre.size == 0 or post.size == 0:
        return None
    y_t_post = _cohort_mean(df, gt, post.min(), post.max(), outcome, time)
    y_t_pre = _cohort_mean(df, gt, pre.min(), pre.max(), outcome, time)
    y_c_post = _cohort_mean(df, gc, post.min(), post.max(), outcome, time)
    y_c_pre = _cohort_mean(df, gc, pre.min(), pre.max(), outcome, time)
    if any(np.isnan([y_t_post, y_t_pre, y_c_post, y_c_pre])):
        return None
    est = (y_t_post - y_t_pre) - (y_c_post - y_c_pre)
    n_t = int((df["g"] == gt).sum())
    n_c = int((df["g"] == gc).sum())
    w = (n_t + n_c) * (pre.size * post.size)
    return float(w), float(est)
