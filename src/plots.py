"""
plots.py — every figure in the report, as small pure functions.

Each function takes plain arrays/dicts and writes one vector PDF to
``report/figures``. Style rules (kept consistent across all figures):
  * categorical colours in a FIXED order (identity never depends on rank),
  * magnitude -> one-hue sequential ramp; signed residuals -> blue/red
    diverging ramp with a neutral grey midpoint,
  * thin lines, hairline solid grid, recessive axes, one y-axis per panel
    (two measures -> two panels, never a twin axis),
  * a legend whenever a panel has two or more series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from .config import BAKU_BBOX  # noqa: E402

# --- palette (validated categorical order; see README "Figures") -------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#d9d8d2"
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8322f"])

COL_W = 3.5      # IEEE single column (inches)
FULL_W = 7.16    # IEEE double column


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.8, "legend.frameon": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round", "lines.markersize": 3.5,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "pdf.fonttype": 42,
    })


def _save(fig, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _panel_label(ax, text):
    ax.set_title(text, loc="left", fontweight="bold")


def _subplots(ncols, width, height, **kw):
    """Side-by-side panels with constrained layout (labels never collide)."""
    return plt.subplots(1, ncols, figsize=(width, height), layout="constrained", **kw)


def _plain_log_y(ax):
    """Log y-axis with plain numbers (0.2, 0.5, 1) instead of 2x10^-1."""
    ax.set_yscale("log")
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(NullFormatter())


def _baku(lat, lng, *others):
    """Keep points inside the Baku/Absheron box; returns filtered arrays + share kept."""
    lat, lng = np.asarray(lat, float), np.asarray(lng, float)
    ok = (~np.isnan(lat) & ~np.isnan(lng)
          & (lat >= BAKU_BBOX["lat"][0]) & (lat <= BAKU_BBOX["lat"][1])
          & (lng >= BAKU_BBOX["lng"][0]) & (lng <= BAKU_BBOX["lng"][1]))
    has = ~(np.isnan(lat) | np.isnan(lng))
    share = ok.sum() / max(1, has.sum())
    return (lat[ok], lng[ok], *[np.asarray(o)[ok] for o in others], share)


def _map_axes(ax):
    ax.set_xlim(*BAKU_BBOX["lng"])
    ax.set_ylim(*BAKU_BBOX["lat"])
    ax.set_aspect(1 / np.cos(np.radians(40.4)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


# =============================================================================
# EDA
# =============================================================================
def price_distribution(price, out_dir):
    fig, axes = _subplots(2, FULL_W, 2.0)
    hi = np.quantile(price, 0.99)
    axes[0].hist(price[price <= hi] / 1000, bins=60, color=SERIES[0], edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Price (thousand AZN, 99th pct. shown)")
    axes[0].set_ylabel("Listings")
    _panel_label(axes[0], "(a) Price: heavy right tail")
    axes[1].hist(np.log(price), bins=60, color=SERIES[0], edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("log(price in AZN)")
    _panel_label(axes[1], "(b) log(price): close to symmetric")
    return _save(fig, out_dir, "eda_price_distribution")


def area_vs_price(area, price, out_dir):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    hb = ax.hexbin(np.log10(area), np.log10(price), gridsize=45, cmap=SEQ, mincnt=1, bins="log",
                   linewidths=0.1)
    ax.set_xlabel(r"log$_{10}$ area (m$^2$)")
    ax.set_ylabel(r"log$_{10}$ price (AZN)")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Listings (log scale)")
    cb.outline.set_visible(False)
    ax.grid(False)
    return _save(fig, out_dir, "eda_area_vs_price")


def price_by_rooms(rooms, price, out_dir):
    r = np.where(np.isnan(rooms), -1, np.minimum(rooms, 6)).astype(int)
    keys = [k for k in range(1, 7) if np.any(r == k)]
    data = [np.log(price[r == k]) for k in keys]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.0),
                    whiskerprops=dict(color=MUTED, linewidth=0.7),
                    capprops=dict(color=MUTED, linewidth=0.7),
                    boxprops=dict(linewidth=0))
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE_RAMP[2])
    ax.set_xticks(range(1, len(keys) + 1), [f"{k}" if k < 6 else "6+" for k in keys])
    ax.set_xlabel("Rooms")
    ax.set_ylabel("log(price in AZN)")
    for i, k in enumerate(keys, start=1):
        ax.text(i, ax.get_ylim()[0], f"n={np.sum(r == k):,}", ha="center", va="bottom",
                fontsize=5.8, color=INK_2)
    return _save(fig, out_dir, "eda_price_by_rooms")


def price_map(lat, lng, price, out_dir, name="eda_price_map"):
    lat, lng, p, share = _baku(lat, lng, price)
    logp = np.log(p)
    order = np.argsort(logp)
    fig, ax = plt.subplots(figsize=(COL_W, 2.5), layout="constrained")
    sc = ax.scatter(lng[order], lat[order], c=logp[order], s=1.2, cmap=SEQ, linewidths=0,
                    vmin=np.quantile(logp, 0.02), vmax=np.quantile(logp, 0.98), rasterized=True)
    _map_axes(ax)
    ax.set_title(f"Baku / Absheron ({share:.0%} of geo-tagged listings)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("log(price)")
    cb.outline.set_visible(False)
    return _save(fig, out_dir, name)


def location_effect(names, medians, counts, out_dir):
    order = np.argsort(medians)
    fig, ax = plt.subplots(figsize=(COL_W, 0.16 * len(names) + 0.6))
    y = np.arange(len(names))
    ax.barh(y, np.asarray(medians)[order], height=0.6, color=SERIES[0])
    ax.set_yticks(y, [f"{names[i]} ({counts[i]:,})" for i in order])
    ax.set_xlabel("Median price per m² (AZN)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "eda_location_effect")


def premium_share_by_category(names, shares, counts, out_dir):
    order = np.argsort(shares)
    fig, ax = plt.subplots(figsize=(COL_W, 0.2 * len(names) + 0.6))
    y = np.arange(len(names))
    ax.barh(y, np.asarray(shares)[order] * 100, height=0.6, color=SERIES[0])
    ax.axvline(50, color=MUTED, linewidth=0.7)
    ax.set_yticks(y, [f"{names[i]} ({counts[i]:,})" for i in order])
    ax.set_xlabel("Premium listings (%) — train split")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "eda_premium_share")


# =============================================================================
# Decision tree studies
# =============================================================================
def tree_depth_curves(depths, clf_curves, reg_train, reg_val, out_dir, best_clf=None, best_reg=None):
    """clf_curves: {criterion: (train_f1, val_f1)}; reg_*: RMSE(log)."""
    fig, axes = _subplots(2, FULL_W, 2.1)
    ax = axes[0]
    for i, (crit, (tr, va)) in enumerate(clf_curves.items()):
        ax.plot(depths, va, color=SERIES[i], label=f"{crit} — validation")
        ax.plot(depths, tr, color=SERIES[i], linestyle=":", linewidth=1.1, label=f"{crit} — train")
    if best_clf:
        ax.axvline(best_clf, color=MUTED, linewidth=0.7)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("F1 (premium)")
    ax.legend(ncol=2, loc="best")
    _panel_label(ax, "(a) Task B: Gini vs entropy")
    ax = axes[1]
    ax.plot(depths, reg_val, color=SERIES[0], label="validation")
    ax.plot(depths, reg_train, color=SERIES[0], linestyle=":", linewidth=1.1, label="train")
    if best_reg:
        ax.axvline(best_reg, color=MUTED, linewidth=0.7)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("RMSE of log(price)")
    ax.legend(loc="upper right")
    _panel_label(ax, "(b) Task A: MSE criterion")
    return _save(fig, out_dir, "tree_depth_curves")


def tree_param_sweeps(msl_grid, msl_val, mid_grid, mid_val, out_dir):
    """Validation RMSE vs min_samples_leaf and vs min_impurity_decrease (Task A)."""
    fig, axes = _subplots(2, FULL_W, 1.9)
    axes[0].plot(msl_grid, msl_val, marker="o", color=SERIES[0])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("min_samples_leaf (max_depth=None)")
    axes[0].set_ylabel("Val. RMSE log(price)")
    _panel_label(axes[0], "(a) Leaf size regularises a full tree")
    x = np.array(mid_grid, float)
    xplot = np.where(x == 0, np.min(x[x > 0]) / 10, x)
    axes[1].plot(xplot, mid_val, marker="o", color=SERIES[0])
    axes[1].set_xscale("log")
    axes[1].set_xticks(xplot, ["0" if v == 0 else f"{v:g}" for v in x])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("min_impurity_decrease")
    _panel_label(axes[1], "(b) Pre-pruning by impurity decrease")
    return _save(fig, out_dir, "tree_param_sweeps")


def learning_curves(sizes, reg_tr, reg_va, clf_tr, clf_va, out_dir):
    fig, axes = _subplots(2, FULL_W, 1.9)
    axes[0].plot(sizes, reg_va, marker="o", color=SERIES[0], label="validation")
    axes[0].plot(sizes, reg_tr, marker="o", color=SERIES[0], linestyle=":", label="train")
    axes[0].set_ylabel("RMSE log(price)")
    _panel_label(axes[0], "(a) Task A tree")
    axes[1].plot(sizes, clf_va, marker="o", color=SERIES[0], label="validation")
    axes[1].plot(sizes, clf_tr, marker="o", color=SERIES[0], linestyle=":", label="train")
    axes[1].set_ylabel("F1 (premium)")
    _panel_label(axes[1], "(b) Task B tree")
    for ax in axes:
        ax.set_xlabel("Training listings")
        ax.legend()
    return _save(fig, out_dir, "tree_learning_curves")


# =============================================================================
# SVM studies
# =============================================================================
def svm_convergence(curves_lambda, curves_batch, curves_sched, out_dir):
    """Each argument: {label: (epochs, objective)}."""
    fig, axes = _subplots(3, FULL_W, 2.0)
    for ax, curves, title in zip(axes, (curves_lambda, curves_batch, curves_sched),
                                 ("(a) Effect of $\\lambda$", "(b) Batch / averaging",
                                  "(c) Step-size schedule")):
        for i, (label, (ep, J)) in enumerate(curves.items()):
            ax.plot(ep, J, color=SERIES[i], label=label)
        _plain_log_y(ax)
        ax.set_xlabel("Epoch")
        ax.legend()
        _panel_label(ax, title)
    axes[0].set_ylabel(r"Objective $J(\mathbf{w})$")
    return _save(fig, out_dir, "svm_convergence")


def svm_lambda_sweep(lambdas, tr_f1, va_f1, va_auc, margin, sv_frac, out_dir, best=None):
    fig, axes = _subplots(3, FULL_W, 1.9)
    ax = axes[0]
    ax.plot(lambdas, va_f1, marker="o", color=SERIES[0], label="val F1")
    ax.plot(lambdas, tr_f1, marker="o", color=SERIES[0], linestyle=":", label="train F1")
    ax.plot(lambdas, va_auc, marker="s", color=SERIES[1], label="val ROC-AUC")
    ax.set_ylabel("Score")
    ax.legend()
    _panel_label(ax, "(a) Validation metric")
    axes[1].plot(lambdas, margin, marker="o", color=SERIES[0])
    _plain_log_y(axes[1])
    axes[1].set_ylabel(r"Margin $2/\|\mathbf{w}\|$")
    _panel_label(axes[1], "(b) Margin width")
    axes[2].plot(lambdas, np.asarray(sv_frac) * 100, marker="o", color=SERIES[0])
    axes[2].set_ylabel("Support vectors (%)")
    _panel_label(axes[2], "(c) Support vectors")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$")
        if best:
            ax.axvline(best, color=MUTED, linewidth=0.7)
    return _save(fig, out_dir, "svm_lambda_sweep")


def rff_study(gamma_curves, lambdas, D_grid, D_auc, D_err, out_dir):
    """gamma_curves: {gamma: val F1 per lambda}; D_*: per number of features."""
    fig, axes = _subplots(3, FULL_W, 1.9)
    for i, (g, vals) in enumerate(gamma_curves.items()):
        axes[0].plot(lambdas, vals, marker="o", color=SERIES[i], label=f"$\\gamma$={g:g}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("Val. F1")
    axes[0].legend()
    _panel_label(axes[0], "(a) $\\gamma$ and $\\lambda$")
    axes[1].plot(D_grid, D_auc, marker="o", color=SERIES[0])
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Random features D")
    axes[1].set_ylabel("Val. ROC-AUC")
    _panel_label(axes[1], "(b) Number of features")
    axes[2].plot(D_grid, D_err, marker="o", color=SERIES[0])
    axes[2].set_xscale("log", base=2)
    _plain_log_y(axes[2])
    axes[2].set_xlabel("Random features D")
    axes[2].set_ylabel(r"Mean $|\phi(x)^\top\phi(z) - k(x,z)|$")
    _panel_label(axes[2], "(c) Kernel approximation")
    return _save(fig, out_dir, "svm_rff_study")


# =============================================================================
# Results / error analysis
# =============================================================================
def pred_vs_actual(y_true, preds: dict, out_dir):
    """preds: {model name: predicted log price} (<= 3 panels)."""
    fig, axes = _subplots(len(preds), FULL_W, 2.2, sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = np.quantile(y_true, [0.001, 0.999])
    for ax, (name, p) in zip(axes, preds.items()):
        ax.hexbin(y_true, p, gridsize=40, cmap=SEQ, mincnt=1, bins="log",
                  extent=(lo, hi, lo, hi), linewidths=0.1)
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual log(price)")
        ax.set_title(name)
        ax.grid(False)
    axes[0].set_ylabel("Predicted log(price)")
    return _save(fig, out_dir, "results_pred_vs_actual")


def roc_pr_curves(curves: dict, out_dir):
    """curves: {model: (fpr, tpr, auc, precision, recall, ap)}"""
    fig, axes = _subplots(2, FULL_W, 2.3)
    for i, (name, (fpr, tpr, auc, prec, rec, ap)) in enumerate(curves.items()):
        axes[0].plot(fpr, tpr, color=SERIES[i], label=f"{name} (AUC {auc:.3f})")
        axes[1].plot(rec, prec, color=SERIES[i], label=f"{name} (AP {ap:.3f})")
    axes[0].plot([0, 1], [0, 1], color=AXIS, linewidth=0.7)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    _panel_label(axes[0], "(a) ROC — test set")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    _panel_label(axes[1], "(b) Precision–recall — test set")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    return _save(fig, out_dir, "results_roc_pr")


def confusion_matrices(mats: dict, out_dir):
    fig, axes = _subplots(len(mats), FULL_W, 1.9)
    for ax, (name, m) in zip(np.atleast_1d(axes), mats.items()):
        m = np.asarray(m)
        ax.imshow(m, cmap=SEQ, vmin=0, vmax=m.max())
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 * m.max() else INK)
        ax.set_xticks([0, 1], ["standard", "premium"])
        ax.set_yticks([0, 1], ["standard", "premium"])
        ax.set_xlabel("Predicted")
        ax.set_title(name)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Actual")
    return _save(fig, out_dir, "results_confusion")


def importance_bars(panels: dict, out_dir, name="feature_importance", top=12):
    """panels: {title: (feature_names, values)} -> horizontal bars of the top-k."""
    fig, axes = _subplots(len(panels), FULL_W, 0.17 * top + 0.7)
    for ax, (title, (names, vals)) in zip(np.atleast_1d(axes), panels.items()):
        vals = np.asarray(vals)
        idx = np.argsort(vals)[-top:]
        ax.barh(range(len(idx)), vals[idx], height=0.6, color=SERIES[0])
        ax.set_yticks(range(len(idx)), [names[i] for i in idx])
        ax.set_title(title)
        ax.set_xlabel("MDI importance (normalised)")
        ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, name)


def svm_weights(names, weights, out_dir, top=16):
    w = np.asarray(weights)
    idx = np.argsort(np.abs(w))[-top:]
    idx = idx[np.argsort(w[idx])]
    fig, ax = plt.subplots(figsize=(COL_W, 0.15 * top + 0.6), layout="constrained")
    colors = [SERIES[0] if v > 0 else SERIES[7] for v in w[idx]]
    ax.barh(range(len(idx)), w[idx], height=0.6, color=colors)
    ax.axvline(0, color=AXIS, linewidth=0.7)
    ax.set_yticks(range(len(idx)), [names[i] for i in idx])
    ax.set_xlabel("Weight (standardised; + means premium)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "svm_weights")


def error_by_group(groups: dict, out_dir):
    """groups: {panel title: (labels, mean abs error, counts)} for MAE(log) of the final tree."""
    fig, axes = _subplots(len(groups), FULL_W, 2.3)
    for ax, (title, (labels, err, counts)) in zip(np.atleast_1d(axes), groups.items()):
        y = np.arange(len(labels))
        ax.barh(y, err, height=0.6, color=SERIES[0])
        ax.set_yticks(y, [f"{l} ({c:,})" for l, c in zip(labels, counts)])
        ax.set_xlabel("Test MAE of log(price)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    return _save(fig, out_dir, "error_by_group")


def residual_by_decile(deciles, mean_resid, out_dir, labels=None):
    fig, ax = plt.subplots(figsize=(COL_W, 1.9), layout="constrained")
    for i, (name, r) in enumerate(mean_resid.items()):
        ax.plot(deciles, r, marker="o", color=SERIES[i], label=name)
    ax.axhline(0, color=AXIS, linewidth=0.7)
    ax.set_xlabel("Decile of true price (test)")
    ax.set_ylabel("Mean residual (log)")
    ax.legend()
    return _save(fig, out_dir, "error_by_price_decile")


def residual_map(lat, lng, resid, out_dir):
    lat, lng, r, share = _baku(lat, lng, resid)
    lim = np.quantile(np.abs(r), 0.98) if len(r) else 1.0
    order = np.argsort(np.abs(r))
    fig, ax = plt.subplots(figsize=(COL_W, 2.5), layout="constrained")
    sc = ax.scatter(lng[order], lat[order], c=r[order], s=2.5, cmap=DIV, linewidths=0,
                    norm=TwoSlopeNorm(0, -lim, lim), rasterized=True)
    _map_axes(ax)
    ax.set_title(f"Test residuals, Baku / Absheron ({share:.0%} of test)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("Residual, log (red = over)")
    cb.outline.set_visible(False)
    return _save(fig, out_dir, "error_residual_map")


def misclassification_vs_distance(bins, err_rate, counts, out_dir):
    fig, ax = plt.subplots(figsize=(COL_W, 1.8), layout="constrained")
    ax.bar(range(len(bins)), np.asarray(err_rate) * 100, width=0.6, color=SERIES[0])
    ax.set_xticks(range(len(bins)), [f"{b}\n(n={c:,})" for b, c in zip(bins, counts)])
    ax.set_xlabel("|log(price) − log(threshold)|")
    ax.set_ylabel("Test error rate (%)")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "error_tier_distance")


# =============================================================================
# Bonus figures
# =============================================================================
def ensemble_curves(rf_k, rf_ours, rf_sk, gb_k, gb_ours, gb_sk, tree_ref, out_dir):
    fig, axes = _subplots(2, FULL_W, 2.0)
    axes[0].plot(rf_k, rf_ours, color=SERIES[0], label="our random forest")
    axes[0].axhline(rf_sk, color=SERIES[1], linewidth=1.0, label="sklearn RandomForest")
    axes[0].axhline(tree_ref[0], color=MUTED, linewidth=1.0, label="our single tree")
    axes[0].set_xlabel("Trees")
    axes[0].set_ylabel("Test RMSE log(price)")
    _panel_label(axes[0], "(a) Bagging + feature subsampling")
    axes[1].plot(gb_k, gb_ours, color=SERIES[0], label="our gradient boosting")
    axes[1].axhline(gb_sk, color=SERIES[1], linewidth=1.0, label="sklearn GradientBoosting")
    axes[1].axhline(tree_ref[0], color=MUTED, linewidth=1.0, label="our single tree")
    axes[1].set_xlabel("Boosting rounds")
    _panel_label(axes[1], "(b) Least-squares boosting")
    for ax in axes:
        ax.legend()
    return _save(fig, out_dir, "bonus_ensembles")


def pca_figure(evr, Z, tier, out_dir):
    fig, axes = _subplots(2, FULL_W, 2.1)
    cum = np.cumsum(evr)
    axes[0].plot(np.arange(1, len(cum) + 1), cum * 100, color=SERIES[0])
    axes[0].set_xlabel("Principal components")
    axes[0].set_ylabel("Cumulative variance (%)")
    _panel_label(axes[0], "(a) Explained variance")
    for i, (lab, name) in enumerate(((0, "standard"), (1, "premium"))):
        m = tier == lab
        axes[1].scatter(Z[m, 0], Z[m, 1], s=1, linewidths=0, color=SERIES[i], alpha=0.35,
                        label=name, rasterized=True)
    lo = np.quantile(Z[:, :2], 0.005, axis=0)
    hi = np.quantile(Z[:, :2], 0.995, axis=0)
    axes[1].set_xlim(lo[0], hi[0])
    axes[1].set_ylim(lo[1], hi[1])
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(markerscale=6)
    _panel_label(axes[1], "(b) Tiers in the first two PCs")
    return _save(fig, out_dir, "bonus_pca")
