from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from haulage.assumptions import NO_ANTICIPATION, PARALLEL_TRENDS, SUTVA, AssumptionSet
from haulage.estimands import ATT
from haulage.estimators.base import BaseEstimator, EstimatorResult
from haulage.estimators.ols import OLS
from haulage.utils.linalg import two_way_demean
from haulage.utils.validation import ensure_panel


@dataclass
class TWFE(BaseEstimator):
    r"""Two-way fixed effects DiD: $y_{it} = \alpha_i + \lambda_t + \tau D_{it} + X_{it}'\beta + u_{it}$.

    Uses iterative within-transform to absorb unit and time FEs, then OLS on the demeaned
    system with two-way cluster-robust SEs (unit, time).

    Args:
        treatment: Column name of the treatment indicator (or continuous intensity).
        outcome: Column name of the outcome.
        unit: Column name of the unit identifier.
        time: Column name of the time identifier.
        covariates: Additional covariates (entered linearly after within-transform).

    References:
        Wooldridge (2002); de Chaisemartin & D'Haultfoeuille (2020).
    """

    treatment: str = "D"
    outcome: str = "y"
    unit: str = "unit"
    time: str = "t"
    covariates: tuple[str, ...] = ()
    name: str = "TWFE"

    @property
    def assumptions(self) -> AssumptionSet:
        return AssumptionSet(
            assumptions=(PARALLEL_TRENDS, NO_ANTICIPATION, SUTVA),
            estimator="TWFE",
        )

    def fit(self, y: Any, *args: Any, **kwargs: Any) -> EstimatorResult:
        r"""Fit TWFE and return an EstimatorResult.

        Args:
            y: DataFrame containing outcome, treatment, unit, time, and covariates.

        Returns:
            EstimatorResult with point = tau, se = two-way cluster SE on (unit, time).
        """
        if not isinstance(y, pd.DataFrame):
            raise TypeError("TWFE.fit expects a pandas DataFrame as first argument")
        df = ensure_panel(y, self.unit, self.time)
        unit_ids = df[self.unit].astype("category").cat.codes.to_numpy()
        time_ids = df[self.time].astype("category").cat.codes.to_numpy()
        y_arr = df[self.outcome].to_numpy(dtype=np.float64)
        cols = [self.treatment, *self.covariates]
        X_raw = df[cols].to_numpy(dtype=np.float64)
        y_tilde = two_way_demean(y_arr, unit_ids, time_ids)
        X_tilde = two_way_demean(X_raw, unit_ids, time_ids)
        ols = OLS(se="twoway", coef_of_interest=0)
        res = ols.fit(
            y_tilde,
            X_tilde,
            clusters=unit_ids,
            cluster_b=time_ids,
            estimand=ATT(),
            names=list(cols),
        )
        return EstimatorResult(
            estimand=ATT(),
            point=res.point,
            se=res.se,
            n=res.n,
            assumptions=self.assumptions,
            diagnostics={
                **res.diagnostics,
                "n_units": int(np.unique(unit_ids).shape[0]),
                "n_periods": int(np.unique(time_ids).shape[0]),
            },
            extra={"ols": res, "treatment": self.treatment},
            name="TWFE",
        )
