"""
decision_tree.py — a CART decision tree built FROM SCRATCH (NumPy only).

Supports BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> (log) price

Design
------
* ``Node`` objects form the tree; ``_grow`` builds it recursively.
* ``_best_split`` is an exact CART search. For a node with m samples it sorts
  every candidate feature once (one ``argsort`` over the whole m x d block),
  then evaluates EVERY threshold of EVERY feature at once with cumulative
  sums:
      classification: running class counts  -> Gini / entropy of each side
      regression:     running sum(y), sum(y^2) -> SSE of each side
  Candidate thresholds are midpoints between consecutive DISTINCT sorted
  values, like scikit-learn.
* Gain is the impurity decrease  I(parent) - sum_c N_c/N * I(c).
  ``min_impurity_decrease`` uses the same WEIGHTED definition as sklearn,
  (N_node / N_total) * gain >= min_impurity_decrease, so matched settings
  mean the same thing in both libraries.
* Ties: the lowest feature index wins, then the lowest threshold. sklearn
  visits features in a random permutation, so with exact ties the two
  libraries may pick different (equally good) splits.
* After fitting, the tree is also flattened into arrays so ``predict`` walks
  all samples down the tree at once instead of one Python call per row.

scikit-learn is NOT used anywhere in this file.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


class Node:
    """One node of the tree."""
    __slots__ = ("feature", "threshold", "left", "right", "value",
                 "n_samples", "impurity", "depth")

    def __init__(self, value, n_samples: int, impurity: float, depth: int):
        self.feature: int | None = None      # feature index to split on (internal)
        self.threshold: float | None = None  # go left if x[feature] <= threshold
        self.left: Node | None = None
        self.right: Node | None = None
        self.value = value                   # class proba vector or mean target
        self.n_samples = n_samples
        self.impurity = impurity
        self.depth = depth

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# --- impurity functions (work on count/probability arrays, last axis = classes)
def gini_from_counts(counts, n):
    p = counts / n
    return 1.0 - np.sum(p * p, axis=-1)


def entropy_from_counts(counts, n):
    p = counts / n
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return -np.sum(p * logp, axis=-1)


class DecisionTree:
    """
    Parameters mirror sklearn's DecisionTreeClassifier / Regressor.

    task : "classification" | "regression"
    criterion : "gini" | "entropy" (classification) or "mse" (regression)
    max_depth : maximum depth (None = grow until other rules stop it)
    min_samples_split : a node needs at least this many samples to be split
    min_samples_leaf : each child must keep at least this many samples
    min_impurity_decrease : split only if (N_t/N) * gain >= this value
    max_features : features tried per split (None=all, int, float fraction,
                   "sqrt", "log2") — used by the random forest bonus
    random_state : seed for the feature subsampling
    """

    def __init__(
        self,
        task: str = "classification",
        criterion: str | None = None,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features=None,
        random_state: int = 42,
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if criterion is None:
            criterion = "gini" if task == "classification" else "mse"
        valid = ("gini", "entropy") if task == "classification" else ("mse",)
        if criterion not in valid:
            raise ValueError(f"criterion {criterion!r} invalid for {task}; use {valid}")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.root: Node | None = None

    # ------------------------------------------------------------------ impurity
    def _impurity(self, y) -> float:
        """Impurity of a single node from its targets (y = class ids or values)."""
        if self.task == "regression":
            return float(np.var(y)) if len(y) else 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        f = gini_from_counts if self.criterion == "gini" else entropy_from_counts
        return float(f(counts, len(y)))

    def _impurity_from_counts(self, counts, n):
        f = gini_from_counts if self.criterion == "gini" else entropy_from_counts
        return f(counts, n)

    # ------------------------------------------------------------- split search
    def _n_features_to_try(self, d: int) -> int:
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return max(1, min(d, int(mf)))

    def _split_scores(self, X, y, features):
        """
        Gain of every (feature, position) pair for the given candidate features.
        Returns (gain[f, i], sorted_values[i, f]) where position i means
        'left child = the i+1 smallest samples'. Invalid positions get -inf.
        """
        m = X.shape[0]
        Xf = X[:, features]
        order = np.argsort(Xf, axis=0, kind="stable")
        xs = np.take_along_axis(Xf, order, axis=0)                 # (m, f)
        n_left = np.arange(1, m, dtype=float)[:, None]             # (m-1, 1)
        n_right = m - n_left

        if self.task == "classification":
            onehot = np.eye(self.n_classes_)[y]                     # (m, K)
            left = np.cumsum(onehot[order], axis=0)[:-1]            # (m-1, f, K)
            total = onehot.sum(axis=0)
            right = total - left
            imp_l = self._impurity_from_counts(left, n_left[..., None])
            imp_r = self._impurity_from_counts(right, n_right[..., None])
            parent = self._impurity_from_counts(total, m)
            child = (n_left * imp_l + n_right * imp_r) / m
        else:
            yc = y - y.mean()                                       # centred: stable sums
            ys = yc[order]
            s_l = np.cumsum(ys, axis=0)[:-1]
            q_l = np.cumsum(ys * ys, axis=0)[:-1]
            s_t, q_t = yc.sum(), (yc * yc).sum()
            sse_l = q_l - s_l * s_l / n_left
            sse_r = (q_t - q_l) - (s_t - s_l) ** 2 / n_right
            parent = q_t / m
            child = (sse_l + sse_r) / m

        gain = parent - child                                       # (m-1, f)
        msl = self.min_samples_leaf
        valid = xs[1:] > xs[:-1]                                    # distinct neighbours
        valid &= (n_left >= msl) & (n_right >= msl)
        gain = np.where(valid, gain, -np.inf)
        return gain.T, xs

    def _best_split(self, X, y):
        """
        Search features & thresholds; return (feature, threshold, gain) of the
        split with the largest impurity decrease, or None if no valid split.
        """
        d = X.shape[1]
        k = self._n_features_to_try(d)
        if k < d:
            perm = self._rng.permutation(d)
            groups = [np.sort(perm[:k]), np.sort(perm[k:])]   # fall back to the rest
        else:
            groups = [np.arange(d)]
        for features in groups:
            gain, xs = self._split_scores(X, y, features)
            flat = int(np.argmax(gain))                 # feature-major: lowest index wins ties
            j, i = divmod(flat, gain.shape[1])
            if np.isfinite(gain[j, i]):
                lo, hi = xs[i, j], xs[i + 1, j]
                thr = (lo + hi) / 2.0
                if thr >= hi:                           # float rounding guard (as sklearn)
                    thr = lo
                return int(features[j]), float(thr), float(gain[j, i])
        return None

    # ------------------------------------------------------------- grow / fit
    def _leaf_value(self, y):
        if self.task == "regression":
            return float(np.mean(y))
        return np.bincount(y, minlength=self.n_classes_) / len(y)

    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        m = len(y)
        impurity = self._impurity(y)
        node = Node(self._leaf_value(y), m, impurity, depth)
        self.max_depth_reached_ = max(self.max_depth_reached_, depth)

        if ((self.max_depth is not None and depth >= self.max_depth)
                or m < self.min_samples_split
                or m < 2 * self.min_samples_leaf
                or impurity <= _EPS):
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, gain = split
        weighted_gain = m / self._n_total * gain
        if weighted_gain + _EPS < self.min_impurity_decrease:
            return node

        self._importances[feature] += weighted_gain
        mask = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with one row per target")
        if self.task == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
        else:
            y = y.astype(float)
        self.n_features_in_ = X.shape[1]
        self._n_total = len(y)
        self._rng = np.random.default_rng(self.random_state)
        self._importances = np.zeros(X.shape[1])
        self.max_depth_reached_ = 0
        self.root = self._grow(X, y, depth=0)
        total = self._importances.sum()
        self.feature_importances_ = self._importances / total if total > 0 else self._importances
        self._flatten()
        return self

    def _flatten(self):
        """Store the tree as parallel arrays (index 0 = root) for fast predict."""
        feats, thrs, lefts, rights, values, depths, ns = [], [], [], [], [], [], []

        def visit(node: Node) -> int:
            i = len(feats)
            feats.append(-1 if node.is_leaf() else node.feature)
            thrs.append(np.nan if node.is_leaf() else node.threshold)
            lefts.append(-1)
            rights.append(-1)
            values.append(node.value)
            depths.append(node.depth)
            ns.append(node.n_samples)
            if not node.is_leaf():
                lefts[i] = visit(node.left)
                rights[i] = visit(node.right)
            return i

        visit(self.root)
        self.tree_feature_ = np.array(feats)
        self.tree_threshold_ = np.array(thrs)
        self.tree_left_ = np.array(lefts)
        self.tree_right_ = np.array(rights)
        self.tree_value_ = np.array(values, dtype=float)
        self.tree_depth_ = np.array(depths)
        self.tree_n_samples_ = np.array(ns)

    # ------------------------------------------------------------- predict
    def _predict_one(self, x, node: Node):
        """Readable reference traversal for one sample (used in the tests)."""
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def apply(self, X, truncate_depth: int | None = None) -> np.ndarray:
        """
        Index of the node each row ends in. With ``truncate_depth=d`` the walk
        stops at depth d, which gives EXACTLY the predictions of a tree fitted
        with max_depth=d (greedy splits only depend on the node's own data) —
        used to draw depth curves from one fit (verified in the tests).
        """
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        limit = np.inf if truncate_depth is None else truncate_depth
        active = (self.tree_feature_[node] >= 0) & (self.tree_depth_[node] < limit)
        while active.any():
            idx = np.flatnonzero(active)
            nd = node[idx]
            go_left = X[idx, self.tree_feature_[nd]] <= self.tree_threshold_[nd]
            node[idx] = np.where(go_left, self.tree_left_[nd], self.tree_right_[nd])
            nd = node[idx]
            active[idx] = (self.tree_feature_[nd] >= 0) & (self.tree_depth_[nd] < limit)
        return node

    def predict(self, X, truncate_depth: int | None = None):
        leaf = self.apply(X, truncate_depth)
        if self.task == "regression":
            return self.tree_value_[leaf]
        return self.classes_[np.argmax(self.tree_value_[leaf], axis=1)]

    def predict_proba(self, X, truncate_depth: int | None = None):
        """Classification only: class frequencies of the training rows in the leaf."""
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification")
        return self.tree_value_[self.apply(X, truncate_depth)]

    # ------------------------------------------------------------- inspection
    def get_depth(self) -> int:
        return int(self.tree_depth_.max())

    def get_n_leaves(self) -> int:
        return int(np.sum(self.tree_feature_ < 0))

    @property
    def node_count(self) -> int:
        return len(self.tree_feature_)

    def export_text(self, feature_names=None, max_depth: int = 3, decimals: int = 3) -> str:
        """Human-readable dump of the top of the tree."""
        names = feature_names or [f"x{i}" for i in range(self.n_features_in_)]
        lines = []

        def fmt(v):
            if self.task == "regression":
                return f"value={v:.{decimals}f}"
            return f"class={self.classes_[int(np.argmax(v))]} p={np.round(v, decimals).tolist()}"

        def rec(node: Node, indent: str):
            if node.is_leaf() or node.depth >= max_depth:
                lines.append(f"{indent}|--- {fmt(node.value)} (n={node.n_samples})")
                return
            name = names[node.feature]
            lines.append(f"{indent}|--- {name} <= {node.threshold:.{decimals}f}")
            rec(node.left, indent + "|   ")
            lines.append(f"{indent}|--- {name} >  {node.threshold:.{decimals}f}")
            rec(node.right, indent + "|   ")

        rec(self.root, "")
        return "\n".join(lines)
