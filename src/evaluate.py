"""
evaluate.py — metrics + comparison helpers, implemented FROM SCRATCH in NumPy.

scikit-learn is used only in the tests, to check these numbers.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))


def regression_metrics(log_true, log_pred) -> dict:
    """
    Metrics on log(price) (what the models fit) AND on price in AZN
    (exp back-transform; that predicts the conditional median, which is the
    honest choice for a heavy-tailed target).
    """
    p_true, p_pred = np.exp(log_true), np.exp(log_pred)
    return {
        "rmse_log": rmse(log_true, log_pred), "mae_log": mae(log_true, log_pred),
        "r2_log": r2(log_true, log_pred),
        "rmse_azn": rmse(p_true, p_pred), "mae_azn": mae(p_true, p_pred),
        "r2_azn": r2(p_true, p_pred), "mape": mape(p_true, p_pred),
    }


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]] (positive class = 1)."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def rankdata(a) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (like scipy 'average')."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    boundary = np.r_[True, s[1:] != s[:-1]]
    group = np.cumsum(boundary) - 1
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], len(a)]
    avg = (starts + 1 + ends) / 2.0              # mean of ranks start+1 .. end
    ranks = np.empty(len(a))
    ranks[order] = avg[group]
    return ranks


def roc_auc(y_true, scores) -> float:
    """
    AUC = P(score of a random positive > score of a random negative), ties
    counted as 1/2 — the Mann-Whitney U statistic computed from ranks.
    """
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _binary_clf_curve(y_true, scores):
    """Cumulative TP/FP counts at each distinct threshold (descending scores)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], y_true[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tps = np.cumsum(y)[last]
    fps = (last + 1) - tps
    return fps, tps, s[last]


def roc_curve(y_true, scores):
    """Return (fpr, tpr, thresholds), starting at (0, 0)."""
    fps, tps, thr = _binary_clf_curve(y_true, scores)
    fpr = np.r_[0.0, fps / fps[-1]] if fps[-1] else np.r_[0.0, fps * 0.0]
    tpr = np.r_[0.0, tps / tps[-1]] if tps[-1] else np.r_[0.0, tps * 0.0]
    return fpr, tpr, np.r_[np.inf, thr]


def precision_recall_curve(y_true, scores):
    """Return (precision, recall) at every distinct threshold (descending)."""
    fps, tps, _ = _binary_clf_curve(y_true, scores)
    precision = tps / (tps + fps)
    recall = tps / tps[-1] if tps[-1] else tps * 0.0
    return precision, recall


def average_precision(y_true, scores) -> float:
    """PR-AUC as average precision: sum_n (R_n - R_{n-1}) P_n (sklearn's definition)."""
    precision, recall = precision_recall_curve(y_true, scores)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def classification_metrics(y_true, y_pred, scores) -> dict:
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred), "precision": p, "recall": r, "f1": f1,
        "roc_auc": roc_auc(y_true, scores), "pr_auc": average_precision(y_true, scores),
    }


# --- Uncertainty --------------------------------------------------------------
def bootstrap_ci(metric, *arrays, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Percentile bootstrap (1 - alpha) CI of ``metric(*arrays)`` over test rows."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = metric(*(np.asarray(a)[idx] for a in arrays))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mean_std(dicts: list[dict]) -> dict:
    """{metric: (mean, std)} across CV folds."""
    keys = dicts[0].keys()
    return {k: (float(np.mean([d[k] for d in dicts])), float(np.std([d[k] for d in dicts])))
            for k in keys}


@contextmanager
def timer(store: dict, key: str):
    """with timer(times, 'fit'): ...  -> times['fit'] = seconds."""
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


# --- Comparison helper ------------------------------------------------------
def format_table(rows: list[dict], columns: list[str], title: str = "") -> str:
    """Fixed-width text table; floats get 4 decimals, (mean, std) -> 'm ± s'."""
    def fmt(v):
        if isinstance(v, tuple) and len(v) == 2:
            return f"{v[0]:.4f} ± {v[1]:.4f}"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        return str(v)

    cells = [[fmt(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    line = "-+-".join("-" * w for w in widths)
    out = [title] if title else []
    out.append(" | ".join(c.ljust(w) for c, w in zip(columns, widths)))
    out.append(line)
    out += [" | ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(out)


def compare(name_to_metrics: dict, columns: list[str] | None = None, title: str = "") -> str:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    ``name_to_metrics`` maps model name -> {metric: value or (mean, std)}.
    Returns the table string (also printed).
    """
    if columns is None:
        columns = list(next(iter(name_to_metrics.values())).keys())
    rows = [{"model": name, **m} for name, m in name_to_metrics.items()]
    table = format_table(rows, ["model"] + columns, title)
    print(table)
    return table
