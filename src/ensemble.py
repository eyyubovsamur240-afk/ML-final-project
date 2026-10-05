"""
ensemble.py — BONUS (Problem 9): ensembles built from OUR OWN DecisionTree.

* RandomForest  — bootstrap rows + feature subsampling at every split
                  (``max_features``), average of the trees' outputs.
* GradientBoostingRegressor — least-squares boosting (Friedman, 2001):
  start from mean(y), then repeatedly fit a shallow regression tree to the
  current residuals and add ``learning_rate`` x its prediction.

Both expose ``staged_predict`` so we can plot test error as learners are
added.
"""

from __future__ import annotations

import numpy as np

from .decision_tree import DecisionTree


class RandomForest:
    def __init__(self, task: str = "classification", n_estimators: int = 50,
                 criterion: str | None = None, max_depth: int | None = None,
                 min_samples_leaf: int = 1, max_features="sqrt", bootstrap: bool = True,
                 random_state: int = 42):
        self.task = task
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.trees_ = []
        n = len(y)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tree = DecisionTree(task=self.task, criterion=self.criterion,
                                max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features=self.max_features,
                                random_state=int(rng.integers(2**31 - 1)))
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        imp = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    def _tree_output(self, tree, X):
        if self.task == "regression":
            return tree.predict(X)
        # align each tree's classes with the forest's (a bootstrap may miss one)
        proba = np.zeros((len(X), len(self.classes_)))
        cols = np.searchsorted(self.classes_, tree.classes_)
        proba[:, cols] = tree.predict_proba(X)
        return proba

    def staged_output(self, X):
        """Yield the ensemble output (mean value / mean proba) after 1, 2, ... trees."""
        total = None
        for k, tree in enumerate(self.trees_, start=1):
            out = self._tree_output(tree, X)
            total = out if total is None else total + out
            yield total / k

    def predict_proba(self, X):
        *_, last = self.staged_output(X)
        return last

    def predict(self, X):
        out = self.predict_proba(X)
        if self.task == "regression":
            return out
        return self.classes_[np.argmax(out, axis=1)]


class GradientBoostingRegressor:
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.1,
                 max_depth: int = 4, min_samples_leaf: int = 1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.init_ = float(y.mean())
        pred = np.full(len(y), self.init_)
        self.trees_ = []
        self.train_loss_ = []
        for _ in range(self.n_estimators):
            residual = y - pred                     # negative gradient of 1/2 (y - f)^2
            tree = DecisionTree(task="regression", criterion="mse", max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                random_state=self.random_state).fit(X, residual)
            pred += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.train_loss_.append(float(np.mean((y - pred) ** 2)))
        imp = np.sum([t.feature_importances_ for t in self.trees_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    def staged_predict(self, X):
        pred = np.full(len(X), self.init_)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict(X)
            yield pred

    def predict(self, X):
        *_, last = self.staged_predict(X)
        return last
