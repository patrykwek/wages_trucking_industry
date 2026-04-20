from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from haulage.utils.linalg import hat_diagonal, xtx_inv


def _meat(X: NDArray[np.float64], u: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    wu = w[:, None] * X * u[:, None]
    out: NDArray[np.float64] = wu.T @ (X * u[:, None])
    return out


def hc0(X: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""White (1980) heteroskedasticity-robust covariance: $V = (X'X)^{-1} X' \Omega X (X'X)^{-1}$
    with $\Omega = \mathrm{diag}(u_i^2)$.
    """
    A = xtx_inv(X)
    B = (X * u[:, None]).T @ (X * u[:, None])
    out: NDArray[np.float64] = A @ B @ A
    return out


def hc1(X: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stata-style small-sample adjustment: HC1 = n / (n - k) * HC0."""
    n, k = X.shape
    out: NDArray[np.float64] = (n / (n - k)) * hc0(X, u)
    return out


def hc2(X: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""MacKinnon & White (1985) HC2: $\Omega_{ii} = u_i^2 / (1 - h_{ii})$."""
    h = hat_diagonal(X)
    w = 1.0 / np.clip(1.0 - h, 1e-12, None)
    A = xtx_inv(X)
    return A @ _meat(X, u, w) @ A


def hc3(X: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""MacKinnon & White (1985) HC3: $\Omega_{ii} = u_i^2 / (1 - h_{ii})^2$.

    Closest to a jackknife; recommended default for small n.
    """
    h = hat_diagonal(X)
    w = 1.0 / np.clip((1.0 - h) ** 2, 1e-12, None)
    A = xtx_inv(X)
    return A @ _meat(X, u, w) @ A
