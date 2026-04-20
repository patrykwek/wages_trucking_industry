"""haulage: reproducible causal inference on U.S. trucking deregulation."""

from __future__ import annotations

from haulage._logging import get_logger
from haulage.assumptions import Assumption, AssumptionSet
from haulage.estimands import ATE, ATT, CATE, LATE, QTE, ATTgt, Estimand, SharpRD
from haulage.estimators.base import BaseEstimator, EstimatorResult
from haulage.influence import InfluenceFunction

__version__ = "0.1.0"

__all__ = [
    "ATE",
    "ATT",
    "CATE",
    "LATE",
    "QTE",
    "ATTgt",
    "Assumption",
    "AssumptionSet",
    "BaseEstimator",
    "Estimand",
    "EstimatorResult",
    "InfluenceFunction",
    "SharpRD",
    "__version__",
    "get_logger",
]
