"""
clustering.py — BONUS (Problem 10): k-means and PCA from scratch (NumPy only).

* KMeans: k-means++ seeding + Lloyd iterations, best of ``n_init`` restarts.
  Used on listing coordinates (projected to km) to find neighbourhood
  clusters, which then sharpen the error analysis.
* PCA: SVD of the centred training matrix.
"""

from __future__ import annotations

import numpy as np


def latlng_to_km(lat, lng, lat0: float = 40.4):
    """Local equirectangular projection: degrees -> km (good enough within Azerbaijan)."""
    lat, lng = np.asarray(lat, float), np.asarray(lng, float)
    return np.column_stack([lng * 111.32 * np.cos(np.radians(lat0)), lat * 110.57])


class KMeans:
    def __init__(self, n_clusters: int = 8, n_init: int = 5, max_iter: int = 300,
                 tol: float = 1e-6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _sq_dist(X, C):
        return np.maximum((X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :], 0)

    def _init_pp(self, X, rng):
        """k-means++: next centre drawn with probability proportional to D(x)^2."""
        centres = [X[rng.integers(len(X))]]
        for _ in range(1, self.n_clusters):
            d2 = self._sq_dist(X, np.array(centres)).min(axis=1)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
            centres.append(X[rng.choice(len(X), p=probs)])
        return np.array(centres)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            C = self._init_pp(X, rng)
            for _ in range(self.max_iter):
                labels = self._sq_dist(X, C).argmin(axis=1)
                newC = np.array([X[labels == k].mean(0) if np.any(labels == k) else C[k]
                                 for k in range(self.n_clusters)])
                shift = np.abs(newC - C).max()
                C = newC
                if shift < self.tol:
                    break
            d2 = self._sq_dist(X, C)
            labels = d2.argmin(axis=1)
            inertia = float(d2[np.arange(len(X)), labels].sum())
            if best is None or inertia < best[0]:
                best = (inertia, C, labels)
        self.inertia_, self.cluster_centers_, self.labels_ = best
        return self

    def predict(self, X):
        return self._sq_dist(np.asarray(X, float), self.cluster_centers_).argmin(axis=1)


class PCA:
    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        var = s ** 2 / (len(X) - 1)
        k = self.n_components or len(s)
        self.components_ = Vt[:k]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum()
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
