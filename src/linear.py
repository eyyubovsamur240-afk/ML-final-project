"""
linear.py — BONUS (Problem 8): Ridge regression from scratch (NumPy only).

Closed form via the normal equations on centred data, so the intercept is
not penalised:

    w_hat = (Xc^T Xc + alpha I)^(-1) Xc^T yc,     b = mean(y) - mean(X) . w_hat

We solve the linear system with ``np.linalg.solve`` instead of forming the
inverse explicitly (cheaper and numerically more stable). alpha = 0 gives
ordinary least squares (falls back to ``lstsq`` if X^T X is singular).
"""

from __future__ import annotations

import numpy as np


class RidgeRegression:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        Xc, yc = X - self.x_mean_, y - self.y_mean_
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        try:
            self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(A, Xc.T @ yc, rcond=None)[0]
        self.intercept_ = float(self.y_mean_ - self.x_mean_ @ self.coef_)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
