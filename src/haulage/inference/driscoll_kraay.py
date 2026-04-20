from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from haulage.inference.hac import _andrews_bandwidth, _kernel_weights
from haulage.utils.linalg import xtx_inv


def driscoll_kraay(
    X: NDArray[np.float64],
    u: NDArray[np.float64],
    time: NDArray[np.int64],
    lags: int | None = None,
    kernel: str = "bartlett",
) -> NDArray[np.float64]:
    r"""Driscoll-Kraay (1998) panel HAC robust to cross-sectional dependence.

    Step 1: sum cross-sectional scores within each time period, $h_t = \sum_i x_{it} u_{it}$.
    Step 2: apply Newey-West HAC to the time series $\{h_t\}$.

    References:
        Driscoll & Kraay (1998), Hoechle (2007).
    """
    labels, inv = np.unique(time, return_inverse=True)
    T = labels.shape[0]
    k = X.shape[1]
    h = np.zeros((T, k), dtype=np.float64)
    np.add.at(h, inv, X * u[:, None])
    L = _andrews_bandwidth(h) if lags is None else lags
    weights = _kernel_weights(kernel, L)
    S = h.T @ h
    for l in range(1, L + 1):
        Gamma_l = h[l:].T @ h[: T - l]
        S += weights[l] * (Gamma_l + Gamma_l.T)
    A = xtx_inv(X)
    return A @ S @ A
