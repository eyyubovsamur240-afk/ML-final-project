"""
tree_compare.py — node-by-node comparison of OUR tree with a fitted sklearn tree.

Two correct CART implementations can still produce different trees because
  (1) TIES: several splits reach exactly the same best gain; sklearn breaks
      ties by visiting features in a random order, we take the lowest index;
  (2) FLOAT THRESHOLDS: sklearn casts X to float32, so distinct float64
      values can collapse into one, which changes the candidate thresholds.

``compare_tree_structure`` walks both trees in lockstep on the training data
and classifies every node, so the benchmark can say *why* the trees differ
instead of just *that* they differ. It only reads ``sk.tree_`` (no sklearn
import), so it is not part of either model.
"""

from __future__ import annotations

import numpy as np


def compare_tree_structure(ours, sk, X, y) -> dict:
    """
    Returns counts of:
      same_split         same feature and the same partition of the node's rows
      same_partition     different feature/threshold but the same partition
      tie_divergence     different partition with EQUAL gain (a tie) -> stop
      float32_divergence different gain, but sklearn's split is exactly the best split
                         our own search finds on the float32-rounded node data (distinct
                         float64 values merged by sklearn's cast) -> stop
      gain_mismatch      different gain that float32 does NOT explain -> stop
                         (this is the number that must be 0 for a correct tree)
      decision_mismatch  one tree splits, the other makes a leaf
      leaves_matched     both trees make a leaf on the same rows
    plus the largest threshold difference among ``same_split`` nodes.
    """
    X = np.asarray(X, dtype=float)
    # sklearn stores X as float32 but compares x <= threshold in double precision.
    # Upcast explicitly: comparing a float32 array with a float64 scalar would be done
    # in float32 under NumPy 1.x value-based casting and shift rows across the threshold.
    X32 = X.astype(np.float32).astype(np.float64)
    t = sk.tree_
    if ours.task == "classification":
        y = np.searchsorted(ours.classes_, np.asarray(y))
    else:
        y = np.asarray(y, dtype=float)

    def gain(idx, left):
        n = len(idx)
        nl = int(left.sum())
        if nl in (0, n):
            return 0.0
        yl, yr = y[idx][left], y[idx][~left]
        return (ours._impurity(y[idx])
                - nl / n * ours._impurity(yl) - (n - nl) / n * ours._impurity(yr))

    stats = dict(nodes_compared=0, same_split=0, same_partition=0, tie_divergence=0,
                 float32_divergence=0, gain_mismatch=0, decision_mismatch=0, leaves_matched=0,
                 max_threshold_diff=0.0)
    stack = [(ours.root, 0, np.arange(len(y)))]
    while stack:
        node, sid, idx = stack.pop()
        stats["nodes_compared"] += 1
        sk_leaf = t.children_left[sid] == -1
        if node.is_leaf() or sk_leaf:
            stats["leaves_matched" if node.is_leaf() == sk_leaf else "decision_mismatch"] += 1
            continue
        lo = X[idx, node.feature] <= node.threshold
        ls = X32[idx, t.feature[sid]] <= float(t.threshold[sid])
        if np.array_equal(lo, ls) or np.array_equal(lo, ~ls):
            same_side = np.array_equal(lo, ls)
            if node.feature == t.feature[sid] and same_side:
                stats["same_split"] += 1
                stats["max_threshold_diff"] = max(stats["max_threshold_diff"],
                                                  abs(node.threshold - t.threshold[sid]))
            else:
                stats["same_partition"] += 1
            sk_l, sk_r = t.children_left[sid], t.children_right[sid]
            if not same_side:
                sk_l, sk_r = sk_r, sk_l
            stack.append((node.left, sk_l, idx[lo]))
            stack.append((node.right, sk_r, idx[~lo]))
        elif np.isclose(gain(idx, lo), gain(idx, ls), rtol=1e-9, atol=1e-12):
            stats["tie_divergence"] += 1
        else:
            best32 = ours._best_split(X32[idx], y[idx])
            if best32 is not None and np.isclose(best32[2], gain(idx, ls), rtol=1e-9, atol=1e-12):
                stats["float32_divergence"] += 1
            else:
                stats["gain_mismatch"] += 1
    return stats
