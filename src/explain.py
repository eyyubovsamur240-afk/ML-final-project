"""
explain.py — what drives each final model, and where each one is weakest.

  * Permutation importance on the TEST split, for several models at once.
    Columns are grouped by the raw feature they come from (all one-hot
    columns of ``district`` move together, as do a feature and its
    missing-indicator), so the importance of a categorical feature is not
    split across dozens of columns. A group's importance is how much the
    test score gets worse when its columns are shuffled across rows:
    RMSE(log) increase for Task A, ROC-AUC drop for Task B. Unlike the
    trees' MDI importances this works for every model (ridge, SVMs, forest)
    and is measured on held-out data.
  * Error breakdown by district and by price band (AZN), per model.

Everything is computed from models already fitted on train+val, so no
model is refitted here and the test split is still only scored, not tuned on.
"""

from __future__ import annotations

import numpy as np

from . import config
from . import evaluate as ev
from . import plots

SEED = config.SEED

# Price bands in AZN (left-closed). Fixed edges so the labels mean the same
# thing in every run; empty bands are dropped.
PRICE_BANDS = [0, 75_000, 125_000, 200_000, 350_000, 600_000, np.inf]
MAX_DISTRICTS = 12          # named districts shown; the rest are pooled as "other"
MIN_DISTRICT_ROWS = 30      # a district needs this many TEST rows to be shown on its own
# Columns that carry the same information (a monotone transform) share a group,
# otherwise shuffling one leaves the other as a perfect stand-in.
ALIASES = {"area_m2": "area", "log_area": "area"}


# --------------------------------------------------------------------------- grouping
def feature_groups(names: list[str]) -> dict[str, list[int]]:
    """Map each raw feature to its design-matrix columns.
    'district=x', 'district=other' -> 'district'; 'rooms_missing' -> 'rooms';
    area_m2 and log_area -> 'area'."""
    groups: dict[str, list[int]] = {}
    for j, n in enumerate(names):
        if "=" in n:
            base = n.split("=", 1)[0]
        elif n.endswith("_missing"):
            base = n[: -len("_missing")]
        else:
            base = n
        base = ALIASES.get(base, base)
        groups.setdefault(base, []).append(j)
    return groups


# --------------------------------------------------------------------------- importance
def permutation_importance(score, mats: dict, groups: dict, n_repeats: int = 3,
                           seed: int = SEED, higher_is_better: bool = False) -> dict:
    """
    score(mats) -> float, where ``mats`` is a dict of aligned matrices (for
    example raw X and standardised Z). Every matrix gets the SAME row
    permutation on the group's columns, so models that read different
    matrices see the same shuffled data.

    Returns {group: (mean, std)} of the score degradation (positive = the
    model needed this feature).
    """
    rng = np.random.default_rng(seed)
    base = score(mats)
    n = next(iter(mats.values())).shape[0]
    out = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            shuffled = {}
            for k, M in mats.items():
                M2 = M.copy()
                M2[:, cols] = M[perm][:, cols]
                shuffled[k] = M2
            s = score(shuffled)
            drops.append(base - s if higher_is_better else s - base)
        out[g] = (float(np.mean(drops)), float(np.std(drops)))
    return out


# --------------------------------------------------------------------------- breakdown
def price_band_labels(price, edges=PRICE_BANDS) -> tuple[np.ndarray, list[str]]:
    def k(v):
        return f"{v / 1000:.0f}k"
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0:
            labels.append(f"<{k(hi)}")
        elif np.isinf(hi):
            labels.append(f"{k(lo)}+")
        else:
            labels.append(f"{k(lo)}–{k(hi)}")
    idx = np.digitize(price, edges[1:-1])
    return idx, labels


def district_labels(district, max_n: int = MAX_DISTRICTS, min_rows: int = MIN_DISTRICT_ROWS):
    """Top districts by test count; rarer ones -> 'other', missing -> '(none)'."""
    d = np.array(["(none)" if not isinstance(v, str) else v for v in district], dtype=object)
    names, counts = np.unique(d[d != "(none)"], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    keep = [names[i] for i in order[:max_n] if counts[i] >= min_rows]
    lab = np.where(np.isin(d, keep) | (d == "(none)"), d, "other")
    return lab, keep


def breakdown(labels, order: list[str], y_true, preds: dict, tier_true=None, tier_preds=None) -> list[dict]:
    """Per group: n, RMSE(log) and mean residual per regression model, error
    rate per classifier."""
    rows = []
    for g in order:
        m = labels == g
        if not np.any(m):
            continue
        r = {"group": g, "n": int(m.sum()),
             "rmse": {k: ev.rmse(y_true[m], p[m]) for k, p in preds.items()},
             "bias": {k: float(np.mean(p[m] - y_true[m])) for k, p in preds.items()}}
        if tier_preds:
            r["tier_err"] = {k: float(np.mean(p[m] != tier_true[m])) for k, p in tier_preds.items()}
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- driver
def run(data, s, fin: dict, extra_models: dict | None = None, n_repeats: int = 3) -> dict:
    """
    ``fin`` is experiments.final_models' output. ``extra_models`` may hold
    the bonus forest/boosting models (fitted on the same train+val design):
    {"Our random forest": ("X", model), ...} for Task A.
    """
    D = fin["_design"]
    models = fin["_models"]
    preds = fin["_predictions"]
    y_te, t_te = D.ylog[1], D.tier[1]
    groups = feature_groups(D.names)
    mats = {"X": D.X[1], "Z": D.Z[1]}
    res: dict = {"figures": [], "n_repeats": n_repeats, "n_groups": len(groups)}

    # --- which models -------------------------------------------------------------
    reg = {"Our tree": ("X", models["reg::Ours: decision tree"]),
           "Our ridge": ("Z", models["reg::Ours: ridge (bonus)"])}
    for name, (kind, m) in (extra_models or {}).items():
        reg[name] = (kind, m)
    clf = {"Our tree": ("X", models["clf::Ours: decision tree"]),
           "Our linear SVM": ("Z", models["clf::Ours: linear SVM (Pegasos)"]),
           "Our RFF SVM": ("Z", models["clf::Ours: RBF SVM (RFF + Pegasos)"])}

    from .experiments import clf_scores, log_step  # local import: experiments imports us

    # --- permutation importance ---------------------------------------------------
    imp_reg, imp_clf = {}, {}
    for name, (kind, m) in reg.items():
        imp_reg[name] = permutation_importance(
            lambda M, m=m, kind=kind: ev.rmse(y_te, m.predict(M[kind])), mats, groups, n_repeats)
        log_step(f"  permutation importance [A] {name} done")
    for name, (kind, m) in clf.items():
        imp_clf[name] = permutation_importance(
            lambda M, m=m, kind=kind: ev.roc_auc(t_te, clf_scores(m, M[kind])), mats, groups,
            n_repeats, higher_is_better=True)
        log_step(f"  permutation importance [B] {name} done")
    res["perm_importance_reg"] = imp_reg
    res["perm_importance_clf"] = imp_clf

    def ranked(imp: dict, ref: str) -> list[tuple[str, float]]:
        return sorted(((g, v[0]) for g, v in imp[ref].items()), key=lambda t: -t[1])
    ref_reg = "Our random forest" if "Our random forest" in imp_reg else "Our tree"
    res["ref_reg"] = ref_reg
    res["top_reg"] = ranked(imp_reg, ref_reg)[:10]
    res["top_clf"] = ranked(imp_clf, "Our RFF SVM")[:10]
    # does every model agree on the top feature group?
    res["top1_reg"] = {n: max(v.items(), key=lambda t: t[1][0])[0] for n, v in imp_reg.items()}
    res["top1_clf"] = {n: max(v.items(), key=lambda t: t[1][0])[0] for n, v in imp_clf.items()}
    res["figures"].append(plots.permutation_importance_bars(
        {"Task A: RMSE(log) increase when shuffled": (imp_reg, ref_reg),
         "Task B: ROC-AUC drop when shuffled": (imp_clf, "Our RFF SVM")}, s.figures_dir))

    # --- error breakdown -----------------------------------------------------------
    te_df = data.df.iloc[data.te].reset_index(drop=True)
    reg_preds = {"Our tree": preds["reg::Ours: decision tree"],
                 "Our ridge": preds["reg::Ours: ridge (bonus)"]}
    for name, (kind, m) in (extra_models or {}).items():
        reg_preds[name] = m.predict(mats[kind])
    clf_preds = {"Our tree": preds["clf::Ours: decision tree"][0],
                 "Our RFF SVM": preds["clf::Ours: RBF SVM (RFF + Pegasos)"][0]}

    dlab, keep = district_labels(te_df["district"].to_numpy(dtype=object))
    d_order = keep + [g for g in ("other", "(none)") if np.any(dlab == g)]
    res["by_district"] = breakdown(dlab, d_order, y_te, reg_preds, t_te, clf_preds)
    bidx, blabels = price_band_labels(np.exp(y_te))
    bl = np.array([blabels[i] for i in bidx], dtype=object)
    res["by_price_band"] = breakdown(bl, blabels, y_te, reg_preds, t_te, clf_preds)
    res["overall_rmse"] = {k: ev.rmse(y_te, p) for k, p in reg_preds.items()}
    res["overall_tier_err"] = {k: float(np.mean(p != t_te)) for k, p in clf_preds.items()}
    res["threshold_band"] = blabels[int(np.digitize(fin["threshold_dev"], PRICE_BANDS[1:-1]))]
    # do the models find the same districts hard? mean pairwise Spearman correlation
    # of per-district RMSE over the named districts
    named = [r for r in res["by_district"] if r["group"] in keep]
    names_ = list(reg_preds)
    if len(named) >= 3:
        ranks = {m: ev.rankdata(np.array([r["rmse"][m] for r in named])) for m in names_}
        cors = [float(np.corrcoef(ranks[a], ranks[b])[0, 1])
                for i, a in enumerate(names_) for b in names_[i + 1:]]
        res["district_rank_corr_mean"] = float(np.mean(cors))
        res["district_rank_corr_min"] = float(np.min(cors))
    res["figures"].append(plots.error_breakdown(res["by_district"], res["by_price_band"],
                                                list(reg_preds), s.figures_dir))
    return res
