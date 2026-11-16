"""
A small generalized linear model (GLM) engine written from scratch with NumPy.

Fits log-link Poisson and Gamma GLMs by iteratively reweighted least squares (IRLS), the
same algorithm R's glm() and statsmodels use, and returns coefficients, standard errors,
95% confidence intervals and deviance. Supports an offset (e.g. log exposure) and prior weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GLMResult:
    family: str
    names: list
    coef: np.ndarray
    se: np.ndarray
    dispersion: float
    deviance: float
    null_deviance: float
    n_iter: int

    def table(self) -> pd.DataFrame:
        z = self.coef / self.se
        return pd.DataFrame({
            "coef": self.coef, "se": self.se, "z": z, "p_value": 2 * stats.norm.sf(np.abs(z)),
            "relativity": np.exp(self.coef),
            "rel_low_95": np.exp(self.coef - 1.96 * self.se),
            "rel_high_95": np.exp(self.coef + 1.96 * self.se),
        }, index=self.names)

    def predict(self, X: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        eta = X @ self.coef + (0 if offset is None else offset)
        return np.exp(eta)


def _deviance(family, y, mu, w):
    if family == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2 * np.sum(w * (t - (y - mu)))
    if family == "gamma":
        return 2 * np.sum(w * (-np.log(y / mu) + (y - mu) / mu))
    raise ValueError(family)


def fit_glm(X, y, family="poisson", names=None, offset=None, weights=None, tol=1e-10, max_iter=100) -> GLMResult:
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    w = np.ones(n) if weights is None else np.asarray(weights, float)

    # start from the intercept-only fit
    mu = np.full(n, np.average(y * np.exp(-off) if family == "poisson" else y, weights=w))
    mu = mu * np.exp(off) if family == "poisson" else mu
    eta = np.log(mu)
    beta = np.zeros(p)
    dev_old = np.inf
    for it in range(1, max_iter + 1):
        # log link: d(mu)/d(eta) = mu.  Working weight = prior weight * (dmu/deta)^2 / Var(mu)
        W = w * (mu if family == "poisson" else np.ones(n))
        z = (eta - off) + (y - mu) / mu
        XtW = X.T * W
        beta = np.linalg.solve(XtW @ X, XtW @ z)
        eta = X @ beta + off
        mu = np.exp(eta)
        dev = _deviance(family, y, mu, w)
        if abs(dev - dev_old) < tol * (abs(dev) + 0.1):
            break
        dev_old = dev

    W = w * (mu if family == "poisson" else np.ones(n))
    if family == "poisson":
        phi = 1.0
    else:  # Pearson estimate of the Gamma dispersion
        phi = np.sum(w * ((y - mu) / mu) ** 2) / (n - p)
    cov = phi * np.linalg.inv((X.T * W) @ X)

    # null deviance: intercept + offset only
    if family == "poisson":
        mu0 = np.exp(off) * np.sum(w * y) / np.sum(w * np.exp(off))
    else:
        mu0 = np.full(n, np.average(y, weights=w))
    return GLMResult(family, list(names) if names is not None else [f"x{i}" for i in range(p)],
                     beta, np.sqrt(np.diag(cov)), phi, dev, _deviance(family, y, mu0, w), it)
