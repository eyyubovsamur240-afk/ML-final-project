"""
experiments.py — the stages that ``run_all`` executes, in order.

Protocol (no test-set peeking anywhere):
  1. split cleaned rows into train / validation / test (stratified on
     price deciles, seeded);
  2. STUDY + SELECT every hyperparameter on train -> validation;
  3. REFIT the chosen configurations on train+validation ("dev") with the
     preprocessing and the tier threshold re-learned on dev, and score the
     untouched TEST split once (with bootstrap 95% CIs);
  4. UNCERTAINTY: 5-fold stratified CV on dev, with preprocessing, scaling
     and tier threshold re-fitted inside every fold -> mean ± std;
  5. benchmark against scikit-learn with matched settings, error analysis,
     bonuses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as SkGB
from sklearn.ensemble import RandomForestClassifier as SkRFC
from sklearn.ensemble import RandomForestRegressor as SkRFR
from sklearn.linear_model import Ridge as SkRidge
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config, plots
from . import data_prep as dp
from . import evaluate as ev
from .clustering import PCA, KMeans, latlng_to_km
from .decision_tree import DecisionTree
from .ensemble import GradientBoostingRegressor, RandomForest
from .linear import RidgeRegression
from .svm import PegasosSVM, RandomFourierFeatures, RFFPegasosSVM, rbf_kernel
from .svr import PegasosSVR, RFFPegasosSVR
from .tree_compare import compare_tree_structure

SEED = config.SEED
# The Preprocessor puts the numeric features first, so log(area) has a fixed column.
LOG_AREA_COL = dp.NUMERIC_FEATURES.index("log_area")


class PerM2Target:
    """
    Wraps a regressor so it learns log(price per m²) = log(price) - log(area)
    and adds log(area) back when predicting, so it still outputs log(price).

    A tree predicts a constant per leaf. With the plain target every flat in
    a leaf gets the same price; with this target it gets the same price per
    m², and its exact area scales it. Nothing is estimated here (the offset
    is a feature of the row itself), so it cannot leak. Other attributes
    (feature_importances_, get_depth, ...) are forwarded to the inner model.
    """

    def __init__(self, model, col: int = LOG_AREA_COL):
        self.model = model
        self.col = col

    def offset(self, X):
        return np.asarray(X, dtype=float)[:, self.col] if config.PER_M2_TARGET else 0.0

    def fit(self, X, y):
        self.model.fit(X, np.asarray(y, dtype=float) - self.offset(X))
        return self

    def predict(self, X, **kw):
        return self.model.predict(X, **kw) + self.offset(X)

    def staged_predict(self, X):
        for p in self.model.staged_predict(X):
            yield p + self.offset(X)

    def staged_output(self, X):
        for p in self.model.staged_output(X):
            yield p + self.offset(X)

    def __getattr__(self, name):
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)


# =============================================================================
# Settings & data containers
# =============================================================================
@dataclass
class Settings:
    fast: bool = False
    figures_dir: object = config.FIGURES_DIR
    tree_depths: list = field(default_factory=lambda: list(config.TREE_DEPTHS))
    min_leaf_grid: list = field(default_factory=lambda: list(config.TREE_MIN_LEAF_GRID))
    min_decrease_grid: list = field(default_factory=lambda: list(config.TREE_MIN_DECREASE_GRID))
    svm_lambdas: list = field(default_factory=lambda: list(config.SVM_LAMBDAS))
    svm_epochs: int = config.SVM_EPOCHS
    svm_final_epochs: int = config.SVM_FINAL_EPOCHS
    svm_batch: int = config.SVM_BATCH
    svm_average: bool = config.SVM_AVERAGE
    rff_gammas: list = field(default_factory=lambda: list(config.RFF_GAMMAS))
    rff_lambdas: list = field(default_factory=lambda: list(config.RFF_LAMBDAS))
    rff_components: int = config.RFF_COMPONENTS
    rff_component_grid: list = field(default_factory=lambda: list(config.RFF_COMPONENT_GRID))
    forest_trees: int = config.FOREST_TREES
    boost_trees: int = config.BOOST_TREES
    cv_folds: int = config.CV_FOLDS
    n_boot: int = config.N_BOOT
    svc_max_train: int = 20_000
    perm_repeats: int = 3

    @classmethod
    def make(cls, fast: bool, figures_dir) -> "Settings":
        s = cls(fast=fast, figures_dir=figures_dir)
        if fast:
            f = config.FAST
            s.tree_depths = f["tree_depths"]
            s.min_leaf_grid = [1, 5, 20, 50]
            s.min_decrease_grid = [0.0, 1e-4, 1e-2]
            s.svm_lambdas, s.svm_epochs = f["svm_lambdas"], f["svm_epochs"]
            s.svm_final_epochs = f["svm_final_epochs"]
            s.rff_gammas, s.rff_lambdas = f["rff_gammas"], f["rff_lambdas"]
            s.rff_components, s.rff_component_grid = f["rff_components"], f["rff_component_grid"]
            s.forest_trees, s.boost_trees = f["forest_trees"], f["boost_trees"]
            s.cv_folds, s.n_boot = f["cv_folds"], f["n_boot"]
            s.svc_max_train = 3_000
            s.perm_repeats = 1
        return s


@dataclass
class Data:
    df: pd.DataFrame
    feats: pd.DataFrame
    price: np.ndarray
    strata: np.ndarray
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray
    log: dp.CleaningLog

    @property
    def dev(self) -> np.ndarray:
        return np.sort(np.concatenate([self.tr, self.va]))


def log_step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare_data(path=None, fast: bool = False) -> Data:
    """Load -> clean -> features -> stratified split (all deterministic)."""
    raw = dp.load_raw(path)
    log = dp.CleaningLog()
    df = dp.clean(raw, log)
    if fast and len(df) > config.FAST["max_rows"]:
        df = df.sample(n=config.FAST["max_rows"], random_state=SEED).reset_index(drop=True)
        log.notes["fast_mode_subsample"] = len(df)
    feats, price, _ = dp.make_features(df)
    strata = dp.price_strata(price)
    tr, va, te = dp.split_indices(len(price), stratify=strata, seed=SEED)
    return Data(df, feats, price, strata, tr, va, te, log)


def design(data: Data, fit_idx, *eval_idx):
    """
    Fit preprocessing, scaling and the tier threshold on ``fit_idx`` ONLY and
    apply them to every index set. Element 0 of each list is the fit set.
    """
    pre = dp.Preprocessor().fit(data.feats.iloc[fit_idx])
    idxs = (fit_idx, *eval_idx)
    X = [pre.transform(data.feats.iloc[i]) for i in idxs]
    scaler = dp.Standardizer().fit(X[0])
    Z = [scaler.transform(x) for x in X]
    ylog = [np.log(data.price[i]) for i in idxs]
    tier0, thr = dp.make_tier_label(data.price[fit_idx])
    tiers = [tier0] + [dp.make_tier_label(data.price[i], thr)[0] for i in eval_idx]
    return SimpleNamespace(pre=pre, names=pre.feature_names_, X=X, Z=Z, ylog=ylog,
                           tier=tiers, thr=thr, idx=idxs)


def clf_scores(model, X):
    """Continuous score for ROC/PR: P(premium) for trees, margin for SVMs."""
    if hasattr(model, "predict_proba") and not isinstance(model, (PegasosSVM, RFFPegasosSVM)):
        proba = model.predict_proba(X)
        return proba[:, -1] if proba.ndim == 2 else proba
    return model.decision_function(X)


# =============================================================================
# 1. EDA (training split only, so no modelling choice is informed by test rows)
# =============================================================================
def eda(data: Data, s: Settings) -> dict:
    out = s.figures_dir
    tr = data.tr
    df_tr = data.df.iloc[tr]
    p = data.price[tr]
    figs = [plots.price_distribution(p, out),
            plots.area_vs_price(df_tr["area_m2"].to_numpy(), p, out),
            plots.price_by_rooms(df_tr["rooms"].to_numpy(), p, out),
            plots.price_map(df_tr["lat"].to_numpy(), df_tr["lng"].to_numpy(), p, out)]
    ppm = p / df_tr["area_m2"].to_numpy()
    loc = df_tr["location"].fillna("(missing)")
    vc = loc.value_counts()
    top = vc[vc >= 30].index[:18] if (vc >= 30).sum() >= 5 else vc.index[:12]
    med = [float(np.median(ppm[(loc == l).to_numpy()])) for l in top]
    figs.append(plots.location_effect(list(top), med, [int(vc[l]) for l in top], out))
    tier, thr = dp.make_tier_label(p)
    cat = df_tr["category"].fillna("(missing)")
    cvc = cat.value_counts()
    cats = list(cvc.index[:8])
    share = [float(tier[(cat == c).to_numpy()].mean()) for c in cats]
    figs.append(plots.premium_share_by_category(cats, share, [int(cvc[c]) for c in cats], out))

    logp = np.log(p)
    stats = {
        "n_train_rows": int(len(tr)),
        "price_median": float(np.median(p)), "price_mean": float(np.mean(p)),
        "price_p01": float(np.quantile(p, 0.01)), "price_p99": float(np.quantile(p, 0.99)),
        "price_skew": float(((p - p.mean()) ** 3).mean() / p.std() ** 3),
        "logprice_skew": float(((logp - logp.mean()) ** 3).mean() / logp.std() ** 3),
        "corr_logarea_logprice": float(np.corrcoef(np.log(df_tr["area_m2"]), logp)[0, 1]),
        "ppm_median": float(np.median(ppm)),
        "top_locations": {l: m for l, m in zip(top, med)},
        "premium_share_by_category": dict(zip(cats, share)),
        "category_counts": {c: int(cvc[c]) for c in cats},
        "tier_threshold_train": thr,
    }
    return {"figures": figs, "stats": stats}


# =============================================================================
# 2. Decision-tree studies (train -> validation)
# =============================================================================
def _select(records, key, higher_is_better):
    """Best record; ties -> simpler model (smaller depth, larger leaves)."""
    sign = -1 if higher_is_better else 1
    return min(records, key=lambda r: (round(sign * r[key], 10), r["max_depth"], -r["min_samples_leaf"]))


def tree_studies(data: Data, s: Settings) -> dict:
    D = design(data, data.tr, data.va)
    X_tr, X_va = D.X
    yl_tr, yl_va = D.ylog
    t_tr, t_va = D.tier
    depths = s.tree_depths
    max_d = max(depths)

    # ---- Task B: criterion x min_samples_leaf x depth (depth via truncation)
    clf_records, depth_curves = [], {}
    for crit in ("gini", "entropy"):
        for msl in s.min_leaf_grid:
            tree = DecisionTree("classification", crit, max_depth=max_d, min_samples_leaf=msl,
                                random_state=SEED).fit(X_tr, t_tr)
            tr_f1, va_f1 = [], []
            for d in depths:
                pv = tree.predict(X_va, truncate_depth=d)
                va_auc = ev.roc_auc(t_va, tree.predict_proba(X_va, truncate_depth=d)[:, 1])
                _, _, f1v = ev.precision_recall_f1(t_va, pv)
                _, _, f1t = ev.precision_recall_f1(t_tr, tree.predict(X_tr, truncate_depth=d))
                tr_f1.append(f1t)
                va_f1.append(f1v)
                clf_records.append(dict(criterion=crit, min_samples_leaf=msl, max_depth=d,
                                        val_f1=f1v, train_f1=f1t, val_auc=va_auc,
                                        val_acc=ev.accuracy(t_va, pv)))
            if msl == 1:
                depth_curves[crit] = (tr_f1, va_f1)
    best_clf = _select(clf_records, "val_f1", True)
    log_step(f"  tree Task B best: {best_clf['criterion']}, depth={best_clf['max_depth']}, "
             f"min_leaf={best_clf['min_samples_leaf']}, val F1={best_clf['val_f1']:.4f}")
    crit_summary = {}
    for crit in ("gini", "entropy"):
        r = _select([x for x in clf_records if x["criterion"] == crit], "val_f1", True)
        crit_summary[crit] = r

    # ---- Task A: min_samples_leaf x depth
    reg_records, reg_curve = [], None
    msl_full_val = []
    for msl in s.min_leaf_grid:
        tree = PerM2Target(DecisionTree("regression", "mse", max_depth=max_d, min_samples_leaf=msl,
                                        random_state=SEED)).fit(X_tr, yl_tr)
        tr_r, va_r = [], []
        for d in depths:
            va = ev.rmse(yl_va, tree.predict(X_va, truncate_depth=d))
            trr = ev.rmse(yl_tr, tree.predict(X_tr, truncate_depth=d))
            tr_r.append(trr)
            va_r.append(va)
            reg_records.append(dict(min_samples_leaf=msl, max_depth=d, val_rmse=va, train_rmse=trr))
        if msl == 1:
            reg_curve = (tr_r, va_r)
        msl_full_val.append(va_r[-1])
    best_reg = _select(reg_records, "val_rmse", False)
    log_step(f"  tree Task A best: depth={best_reg['max_depth']}, "
             f"min_leaf={best_reg['min_samples_leaf']}, val RMSE(log)={best_reg['val_rmse']:.4f}")

    # ---- third hyperparameter: min_impurity_decrease at the chosen depth/leaf
    mid_val, mid_nodes = [], []
    for mid in s.min_decrease_grid:
        t = PerM2Target(DecisionTree("regression", "mse", max_depth=best_reg["max_depth"],
                                     min_samples_leaf=best_reg["min_samples_leaf"],
                                     min_impurity_decrease=mid)).fit(X_tr, yl_tr)
        mid_val.append(ev.rmse(yl_va, t.predict(X_va)))
        mid_nodes.append(t.node_count)
    best_mid = float(s.min_decrease_grid[int(np.argmin(np.round(mid_val, 10)))])
    best_reg["min_impurity_decrease"] = best_mid
    best_clf["min_impurity_decrease"] = 0.0

    # ---- learning curves at the chosen configs
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(yl_tr))
    sizes, lc = [], {"reg_tr": [], "reg_va": [], "clf_tr": [], "clf_va": []}
    for frac in config.LEARNING_CURVE_FRACTIONS:
        n = max(50, int(frac * len(perm)))
        idx = perm[:n]
        sizes.append(n)
        r = PerM2Target(DecisionTree("regression", "mse", max_depth=best_reg["max_depth"],
                                     min_samples_leaf=best_reg["min_samples_leaf"],
                                     min_impurity_decrease=best_mid)).fit(X_tr[idx], yl_tr[idx])
        lc["reg_tr"].append(ev.rmse(yl_tr[idx], r.predict(X_tr[idx])))
        lc["reg_va"].append(ev.rmse(yl_va, r.predict(X_va)))
        c = DecisionTree("classification", best_clf["criterion"], max_depth=best_clf["max_depth"],
                         min_samples_leaf=best_clf["min_samples_leaf"]).fit(X_tr[idx], t_tr[idx])
        lc["clf_tr"].append(ev.precision_recall_f1(t_tr[idx], c.predict(X_tr[idx]))[2])
        lc["clf_va"].append(ev.precision_recall_f1(t_va, c.predict(X_va))[2])

    out = s.figures_dir
    figs = [
        plots.tree_depth_curves(depths, depth_curves, reg_curve[0], reg_curve[1], out),
        plots.tree_param_sweeps(s.min_leaf_grid, msl_full_val, s.min_decrease_grid, mid_val, out),
        plots.learning_curves(sizes, lc["reg_tr"], lc["reg_va"], lc["clf_tr"], lc["clf_va"], out),
    ]
    return {
        "best_clf": best_clf, "best_reg": best_reg, "criterion_comparison": crit_summary,
        "min_decrease": {"grid": s.min_decrease_grid, "val_rmse": mid_val, "nodes": mid_nodes},
        "min_leaf_full_depth": {"grid": s.min_leaf_grid, "val_rmse": msl_full_val},
        "depth_curve_reg": {"depths": depths, "train": reg_curve[0], "val": reg_curve[1]},
        "depth_curve_clf": {c: {"train": v[0], "val": v[1]} for c, v in depth_curves.items()},
        "learning_curve": {"sizes": sizes, **lc},
        "figures": figs,
    }


# =============================================================================
# 3. SVM studies (train -> validation, standardised features)
# =============================================================================
def svm_studies(data: Data, s: Settings) -> dict:
    D = design(data, data.tr, data.va)
    Z_tr, Z_va = D.Z
    t_tr, t_va = D.tier
    n = len(t_tr)

    sweep, conv_lambda = [], {}
    for lam in s.svm_lambdas:
        m = PegasosSVM(lambda_=lam, n_epochs=s.svm_epochs, batch_size=s.svm_batch,
                       average=s.svm_average, random_state=SEED).fit(Z_tr, t_tr)
        sv = m.support_vectors(Z_tr, t_tr)
        rec = dict(lambda_=lam, C=1 / (n * lam), w_norm=m.w_norm, margin=m.margin_width,
                   n_sv=int(len(sv)), sv_frac=len(sv) / n,
                   train_f1=ev.precision_recall_f1(t_tr, m.predict(Z_tr))[2],
                   val_f1=ev.precision_recall_f1(t_va, m.predict(Z_va))[2],
                   val_auc=ev.roc_auc(t_va, m.decision_function(Z_va)),
                   val_acc=ev.accuracy(t_va, m.predict(Z_va)),
                   final_objective=m.history_["objective"][-1])
        sweep.append(rec)
        if lam in s.svm_lambdas[::max(1, len(s.svm_lambdas) // 4)]:
            conv_lambda[f"$\\lambda$={lam:g}"] = (m.history_["epoch"], m.history_["objective"])
    best = max(sweep, key=lambda r: (round(r["val_f1"], 10), r["val_auc"], r["lambda_"]))
    lam = best["lambda_"]
    log_step(f"  linear SVM best: lambda={lam:g}, val F1={best['val_f1']:.4f}, "
             f"margin={best['margin']:.3f}, SV={best['sv_frac']:.1%}")

    # convergence: batch size / averaging and learning-rate schedule at the chosen lambda
    conv_batch, conv_sched, timings = {}, {}, {}
    for bs, avg, label in ((1, False, "batch 1, last iterate"),
                           (s.svm_batch, False, f"batch {s.svm_batch}, last iterate"),
                           (s.svm_batch, True, f"batch {s.svm_batch}, averaged"),
                           (None, False, "full batch")):
        t0 = time.perf_counter()
        m = PegasosSVM(lambda_=lam, n_epochs=s.svm_epochs, batch_size=bs, average=avg,
                       random_state=SEED).fit(Z_tr, t_tr)
        timings[label] = time.perf_counter() - t0
        conv_batch[label] = (m.history_["epoch"], m.history_["objective"])
    for sched, eta0 in (("pegasos", None), ("invscaling", 0.5), ("constant", 0.05)):
        kw = {} if eta0 is None else {"eta0": eta0}
        m = PegasosSVM(lambda_=lam, n_epochs=s.svm_epochs, batch_size=s.svm_batch,
                       lr_schedule=sched, average=s.svm_average, random_state=SEED, **kw).fit(Z_tr, t_tr)
        label = "1/(λt)" if sched == "pegasos" else (f"{eta0}/√t" if sched == "invscaling" else f"constant {eta0}")
        conv_sched[label] = (m.history_["epoch"], m.history_["objective"])

    # chosen model: margin, support vectors, weights
    final = PegasosSVM(lambda_=lam, n_epochs=s.svm_final_epochs, batch_size=s.svm_batch,
                       average=s.svm_average, random_state=SEED).fit(Z_tr, t_tr)
    J = final.history_["objective"]

    # ---- random Fourier features: gamma x lambda grid
    rff_grid, gamma_curves = [], {}
    for g in s.rff_gammas:
        rff = RandomFourierFeatures(s.rff_components, g, SEED, dtype=np.float32).fit(Z_tr)
        P_tr, P_va = rff.transform(Z_tr), rff.transform(Z_va)
        vals = []
        for lam_r in s.rff_lambdas:
            m = PegasosSVM(lambda_=lam_r, n_epochs=s.svm_epochs, batch_size=s.svm_batch,
                           average=s.svm_average, random_state=SEED,
                           record_objective=False).fit(P_tr, t_tr)
            f1 = ev.precision_recall_f1(t_va, m.predict(P_va))[2]
            vals.append(f1)
            rff_grid.append(dict(gamma=g, lambda_=lam_r, val_f1=f1,
                                 val_auc=ev.roc_auc(t_va, m.decision_function(P_va))))
        gamma_curves[g] = vals
    best_rff = max(rff_grid, key=lambda r: (round(r["val_f1"], 10), r["val_auc"], r["lambda_"]))
    log_step(f"  RFF SVM best: gamma={best_rff['gamma']:g}, lambda={best_rff['lambda_']:g}, "
             f"val F1={best_rff['val_f1']:.4f}")

    # number of random features + kernel approximation quality
    sub = Z_tr[np.random.default_rng(SEED).choice(n, min(400, n), replace=False)]
    K = rbf_kernel(sub, sub, best_rff["gamma"])
    D_auc, D_err = [], []
    for Dn in s.rff_component_grid:
        m = RFFPegasosSVM(gamma=best_rff["gamma"], n_components=Dn, lambda_=best_rff["lambda_"],
                          n_epochs=s.svm_epochs, batch_size=s.svm_batch, random_state=SEED,
                          average=s.svm_average, record_objective=False).fit(Z_tr, t_tr)
        D_auc.append(ev.roc_auc(t_va, m.decision_function(Z_va)))
        # Monte-Carlo error of the kernel approximation, averaged over 5 random draws
        errs = []
        for rep in range(5):
            Phi = RandomFourierFeatures(Dn, best_rff["gamma"], SEED + rep).fit(sub).transform(sub)
            errs.append(float(np.abs(Phi @ Phi.T - K).mean()))
        D_err.append(float(np.mean(errs)))

    out = s.figures_dir
    figs = [
        plots.svm_convergence(conv_lambda, conv_batch, conv_sched, out),
        plots.svm_lambda_sweep([r["lambda_"] for r in sweep], [r["train_f1"] for r in sweep],
                               [r["val_f1"] for r in sweep], [r["val_auc"] for r in sweep],
                               [r["margin"] for r in sweep], [r["sv_frac"] for r in sweep],
                               out, best=lam),
        plots.rff_study(gamma_curves, s.rff_lambdas, s.rff_component_grid, D_auc, D_err, out),
    ]
    return {
        "sweep": sweep, "best": best, "best_rff": best_rff, "rff_grid": rff_grid,
        "rff_components": {"D": s.rff_component_grid, "val_auc": D_auc, "kernel_err": D_err},
        "convergence": {"objective_first": J[0], "objective_last": J[-1],
                        "rel_change_last_5_epochs": float(abs(J[-1] - J[-6]) / J[-1]) if len(J) > 6 else None,
                        "batch_timings_s": timings,
                        "batch_final_objective": {k: v[1][-1] for k, v in conv_batch.items()},
                        "schedule_final_objective": {k: v[1][-1] for k, v in conv_sched.items()}},
        "figures": figs,
    }


# =============================================================================
# 4. Model factories (ours + matched sklearn baselines)
# =============================================================================
def reg_models(tree_cfg, ridge_alpha):
    kw = dict(max_depth=tree_cfg["max_depth"], min_samples_leaf=tree_cfg["min_samples_leaf"],
              min_impurity_decrease=tree_cfg["min_impurity_decrease"])
    return {
        "Ours: decision tree": ("X", lambda: PerM2Target(DecisionTree("regression", "mse", random_state=SEED, **kw))),
        "sklearn DecisionTreeRegressor": ("X", lambda: PerM2Target(DecisionTreeRegressor(random_state=SEED, **kw))),
        "Ours: ridge (bonus)": ("Z", lambda: RidgeRegression(ridge_alpha)),
        "sklearn Ridge": ("Z", lambda: SkRidge(alpha=ridge_alpha)),
    }


def clf_models(tree_cfg, svm_lam, rff_cfg, s: Settings, n_fit: int):
    kw = dict(max_depth=tree_cfg["max_depth"], min_samples_leaf=tree_cfg["min_samples_leaf"])
    crit = tree_cfg["criterion"]
    C = 1.0 / (n_fit * svm_lam)
    C_rff = 1.0 / (n_fit * rff_cfg["lambda_"])
    return {
        "Ours: decision tree": ("X", lambda: DecisionTree("classification", crit, random_state=SEED, **kw)),
        "sklearn DecisionTreeClassifier": ("X", lambda: DecisionTreeClassifier(criterion=crit, random_state=SEED, **kw)),
        "Ours: linear SVM (Pegasos)": ("Z", lambda: PegasosSVM(lambda_=svm_lam, n_epochs=s.svm_final_epochs,
                                                               batch_size=s.svm_batch, random_state=SEED,
                                                               average=s.svm_average,
                                                               record_objective=False)),
        "sklearn SGDClassifier (hinge)": ("Xp", lambda: make_pipeline(
            StandardScaler(), SGDClassifier(loss="hinge", penalty="l2", alpha=svm_lam,
                                            learning_rate="optimal", max_iter=s.svm_epochs,
                                            tol=None, random_state=SEED))),
        "sklearn LinearSVC": ("Xp", lambda: make_pipeline(
            StandardScaler(), LinearSVC(C=C, loss="hinge", dual=True, max_iter=200_000,
                                        random_state=SEED))),
        "Ours: RBF SVM (RFF + Pegasos)": ("Z", lambda: RFFPegasosSVM(
            gamma=rff_cfg["gamma"], n_components=s.rff_components, lambda_=rff_cfg["lambda_"],
            n_epochs=s.svm_epochs, batch_size=s.svm_batch, random_state=SEED,
            average=s.svm_average, record_objective=False)),
        "sklearn SVC (RBF)": ("Xp", lambda: make_pipeline(
            StandardScaler(), SVC(kernel="rbf", gamma=rff_cfg["gamma"], C=C_rff))),
    }


def _matrix(D, kind, j):
    """'X' raw design matrix, 'Z' our standardised matrix, 'Xp' raw for sklearn Pipelines."""
    return D.Z[j] if kind == "Z" else D.X[j]


def _fit_eval_reg(make, kind, D, j_eval=1):
    times = {}
    model = make()
    with ev.timer(times, "fit_s"):
        model.fit(_matrix(D, kind, 0), D.ylog[0])
    with ev.timer(times, "predict_s"):
        pred = model.predict(_matrix(D, kind, j_eval))
    return model, pred, times


def _fit_eval_clf(make, kind, D, j_eval=1, max_train=None):
    times = {}
    model = make()
    X0, y0 = _matrix(D, kind, 0), D.tier[0]
    note = None
    est = model[-1] if hasattr(model, "steps") else model
    if max_train and len(y0) > max_train and type(est).__name__ == "SVC":
        idx = np.random.default_rng(SEED).choice(len(y0), max_train, replace=False)
        X0, y0 = X0[idx], y0[idx]
        note = f"trained on a {max_train:,}-row subsample (O(n^2) kernel solver)"
    with ev.timer(times, "fit_s"):
        model.fit(X0, y0)
    Xe = _matrix(D, kind, j_eval)
    with ev.timer(times, "predict_s"):
        pred = model.predict(Xe)
        score = clf_scores(model, Xe)
    return model, pred, score, times, note


# =============================================================================
# 5. Final models: refit on dev (train+val), score TEST once
# =============================================================================
def select_ridge_alpha(data: Data) -> tuple[float, list]:
    D = design(data, data.tr, data.va)
    rows = []
    for a in config.RIDGE_ALPHAS:
        m = RidgeRegression(a).fit(D.Z[0], D.ylog[0])
        rows.append(dict(alpha=a, val_rmse=ev.rmse(D.ylog[1], m.predict(D.Z[1]))))
    best = min(rows, key=lambda r: (round(r["val_rmse"], 10), -r["alpha"]))
    return best["alpha"], rows


def final_models(data: Data, s: Settings, tree_res, svm_res, ridge_alpha) -> dict:
    D = design(data, data.dev, data.te)
    y_te, t_te = D.ylog[1], D.tier[1]
    res = {"threshold_dev": D.thr, "n_dev": len(data.dev), "n_test": len(data.te),
           "regression": {}, "classification": {}, "_predictions": {}, "notes": {}}
    const = np.full(len(y_te), np.median(D.ylog[0]))
    res["regression"]["Constant baseline (train median)"] = {
        "metrics": ev.regression_metrics(y_te, const), "times": {"fit_s": 0.0, "predict_s": 0.0}}
    models = {}
    for name, (kind, make) in reg_models(tree_res["best_reg"], ridge_alpha).items():
        model, pred, times = _fit_eval_reg(make, kind, D)
        m = ev.regression_metrics(y_te, pred)
        m["rmse_log_ci"] = ev.bootstrap_ci(ev.rmse, y_te, pred, n_boot=s.n_boot, seed=SEED)
        m["r2_log_ci"] = ev.bootstrap_ci(ev.r2, y_te, pred, n_boot=s.n_boot, seed=SEED)
        res["regression"][name] = {"metrics": m, "times": times}
        res["_predictions"][f"reg::{name}"] = pred
        models[f"reg::{name}"] = model
        log_step(f"  [A] {name:34s} RMSE(log)={m['rmse_log']:.4f} R2={m['r2_log']:.4f} "
                 f"fit={times['fit_s']:.2f}s")

    n_fit = len(D.tier[0])
    for name, (kind, make) in clf_models(tree_res["best_clf"], svm_res["best"]["lambda_"],
                                         svm_res["best_rff"], s, n_fit).items():
        model, pred, score, times, note = _fit_eval_clf(make, kind, D, max_train=s.svc_max_train)
        m = ev.classification_metrics(t_te, pred, score)
        m["f1_ci"] = ev.bootstrap_ci(lambda a, b: ev.precision_recall_f1(a, b)[2], t_te, pred,
                                     n_boot=s.n_boot, seed=SEED)
        m["auc_ci"] = ev.bootstrap_ci(ev.roc_auc, t_te, score, n_boot=s.n_boot, seed=SEED)
        m["confusion"] = ev.confusion_matrix(t_te, pred).tolist()
        res["classification"][name] = {"metrics": m, "times": times}
        if note:
            res["notes"][name] = note
        res["_predictions"][f"clf::{name}"] = (pred, score)
        models[f"clf::{name}"] = model
        log_step(f"  [B] {name:34s} F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} "
                 f"fit={times['fit_s']:.2f}s")
    res["paired"] = paired_tests(y_te, t_te, res["_predictions"], s.n_boot)
    res["_models"] = models
    res["_design"] = D
    return res


# Task A tests RMSE only: on a fixed resample R^2 = 1 - MSE / var(y) is a monotone
# function of RMSE, so its paired test gives the same verdict and p-value.
PAIRED_METRICS = {"A": [("rmse_log", ev.rmse)],
                  "B": [("f1", lambda y, p: ev.precision_recall_f1(y, p)[2]), ("roc_auc", ev.roc_auc)]}


def paired_tests(y_log, tier, preds: dict, n_boot: int, pairs=config.PAIRED_COMPARISONS) -> list[dict]:
    """
    Paired bootstrap of every pair in ``pairs`` on the same test rows.
    ``preds`` maps "reg::name" -> log-price predictions and "clf::name" ->
    (labels, scores); F1 uses the labels, ROC-AUC the scores. Pairs with a
    missing model are skipped.
    """
    rows = []
    for task, label, a, b in pairs:
        if a not in preds or b not in preds:
            continue
        for metric, fn in PAIRED_METRICS[task]:
            if task == "A":
                ya, yb, y = preds[a], preds[b], y_log
            else:
                j = 1 if metric == "roc_auc" else 0
                ya, yb, y = preds[a][j], preds[b][j], tier
            r = ev.paired_bootstrap(fn, y, ya, yb, n_boot=n_boot, seed=SEED)
            rows.append({"task": task, "pair": label, "a": a.split("::", 1)[1],
                         "b": b.split("::", 1)[1], "metric": metric, **r})
    return rows


# =============================================================================
# 6. Cross-validated uncertainty on dev
# =============================================================================
def cross_validate(data: Data, s: Settings, tree_res, svm_res, ridge_alpha) -> dict:
    dev = data.dev
    reg_scores, clf_scores_ = {}, {}
    for k, (tr_i, va_i) in enumerate(dp.stratified_kfold(data.strata[dev], s.cv_folds, SEED)):
        D = design(data, dev[tr_i], dev[va_i])
        for name, (kind, make) in reg_models(tree_res["best_reg"], ridge_alpha).items():
            _, pred, times = _fit_eval_reg(make, kind, D)
            m = ev.regression_metrics(D.ylog[1], pred)
            m.update(times)
            reg_scores.setdefault(name, []).append(m)
        for name, (kind, make) in clf_models(tree_res["best_clf"], svm_res["best"]["lambda_"],
                                             svm_res["best_rff"], s, len(tr_i)).items():
            _, pred, score, times, _ = _fit_eval_clf(make, kind, D, max_train=s.svc_max_train)
            m = ev.classification_metrics(D.tier[1], pred, score)
            m.update(times)
            clf_scores_.setdefault(name, []).append(m)
        log_step(f"  CV fold {k + 1}/{s.cv_folds} done")
    return {"regression": {n: ev.mean_std(v) for n, v in reg_scores.items()},
            "classification": {n: ev.mean_std(v) for n, v in clf_scores_.items()},
            "folds": s.cv_folds}


# =============================================================================
# 7. Benchmark details + error analysis on the test split
# =============================================================================
def analysis(data: Data, s: Settings, fin: dict) -> dict:
    D = fin["_design"]
    models = fin["_models"]
    names = D.names
    out = s.figures_dir
    y_te, t_te = D.ylog[1], D.tier[1]
    te_df = data.df.iloc[data.te].reset_index(drop=True)
    res = {"figures": []}

    # --- tree vs sklearn: structure & agreement ---------------------------------
    comp = {}
    for task, ours_key, sk_key, y_fit in (
            ("regression", "reg::Ours: decision tree", "reg::sklearn DecisionTreeRegressor", D.ylog[0]),
            ("classification", "clf::Ours: decision tree", "clf::sklearn DecisionTreeClassifier", D.tier[0])):
        ours, sk = models[ours_key], models[sk_key]
        if isinstance(ours, PerM2Target):       # compare the trees on the target they were fit to
            y_fit = y_fit - ours.offset(D.X[0])
        st = compare_tree_structure(getattr(ours, "model", ours), getattr(sk, "model", sk), D.X[0], y_fit)
        po, ps = ours.predict(D.X[1]), sk.predict(D.X[1])
        agree = float(np.mean(np.isclose(po, ps))) if task == "regression" else float(np.mean(po == ps))
        # float32 effect: distinct float64 values that collapse in float32
        X0 = D.X[0]
        collapsed = int(sum(len(np.unique(X0[:, j])) - len(np.unique(X0[:, j].astype(np.float32)))
                            for j in range(X0.shape[1])))
        imp_corr = float(np.corrcoef(ours.feature_importances_, sk.feature_importances_)[0, 1])
        root_same_partition = None
        if not ours.root.is_leaf() and sk.tree_.node_count > 1:
            lo = X0[:, ours.root.feature] <= ours.root.threshold
            ls = X0.astype(np.float32).astype(np.float64)[:, sk.tree_.feature[0]] <= float(sk.tree_.threshold[0])
            root_same_partition = bool(np.array_equal(lo, ls))
        comp[task] = dict(structure=st, test_agreement=agree, ours_depth=ours.get_depth(),
                          sk_depth=int(sk.get_depth()), ours_leaves=ours.get_n_leaves(),
                          sk_leaves=int(sk.get_n_leaves()), importance_corr=imp_corr,
                          float32_collapsed_values=collapsed, root_same_partition=root_same_partition,
                          root_ours=(names[ours.root.feature], ours.root.threshold) if not ours.root.is_leaf() else None,
                          root_sk=(names[sk.tree_.feature[0]], float(sk.tree_.threshold[0])) if sk.tree_.node_count > 1 else None)
    res["tree_vs_sklearn"] = comp

    # --- figures: predictions, ROC/PR, confusion -------------------------------------
    preds = fin["_predictions"]
    pv = {"Our tree": preds["reg::Ours: decision tree"],
          "sklearn tree": preds["reg::sklearn DecisionTreeRegressor"],
          "Our ridge (bonus)": preds["reg::Ours: ridge (bonus)"]}
    res["figures"].append(plots.pred_vs_actual(y_te, pv, out))
    curves = {}
    for label, key in (("Our tree", "clf::Ours: decision tree"),
                       ("Our linear SVM", "clf::Ours: linear SVM (Pegasos)"),
                       ("Our RFF SVM", "clf::Ours: RBF SVM (RFF + Pegasos)"),
                       ("sklearn tree", "clf::sklearn DecisionTreeClassifier"),
                       ("sklearn SVC (RBF)", "clf::sklearn SVC (RBF)")):
        _, score = preds[key]
        fpr, tpr, _ = ev.roc_curve(t_te, score)
        prec, rec = ev.precision_recall_curve(t_te, score)
        curves[label] = (fpr, tpr, ev.roc_auc(t_te, score), prec, rec, ev.average_precision(t_te, score))
    res["figures"].append(plots.roc_pr_curves(curves, out))
    mats = {lab: ev.confusion_matrix(t_te, preds[key][0]) for lab, key in (
        ("Our tree", "clf::Ours: decision tree"), ("Our linear SVM", "clf::Ours: linear SVM (Pegasos)"),
        ("Our RFF SVM", "clf::Ours: RBF SVM (RFF + Pegasos)"))}
    res["figures"].append(plots.confusion_matrices(mats, out))

    # --- feature importances & SVM weights --------------------------------------------
    tree_reg, tree_clf = models["reg::Ours: decision tree"], models["clf::Ours: decision tree"]
    res["figures"].append(plots.importance_bars({
        "Our tree — Task A (log price)": (names, tree_reg.feature_importances_),
        "Our tree — Task B (tier)": (names, tree_clf.feature_importances_)}, out))
    top = lambda imp: [(names[i], float(imp[i])) for i in np.argsort(imp)[::-1][:10]]
    res["importance_reg"] = top(tree_reg.feature_importances_)
    res["importance_clf"] = top(tree_clf.feature_importances_)
    res["importance_sklearn_reg"] = top(models["reg::sklearn DecisionTreeRegressor"].feature_importances_)
    svm = models["clf::Ours: linear SVM (Pegasos)"]
    res["figures"].append(plots.svm_weights(names, svm.coef_, out))
    order = np.argsort(svm.coef_)
    res["svm_weights_top_pos"] = [(names[i], float(svm.coef_[i])) for i in order[::-1][:8]]
    res["svm_weights_top_neg"] = [(names[i], float(svm.coef_[i])) for i in order[:8]]
    sv = svm.support_vectors(D.Z[0], D.tier[0])
    res["svm_margin"] = dict(w_norm=svm.w_norm, margin=svm.margin_width, n_sv=int(len(sv)),
                             sv_frac=len(sv) / len(D.tier[0]), bias=svm.intercept_)
    # Objective reached by Pegasos vs liblinear's exact solution of the SAME primal:
    # sklearn's StandardScaler == our Standardizer (population std), and liblinear
    # also appends a constant-1 feature for the (regularised) bias, so with
    # C = 1/(n lambda) both minimise  lambda/2 (||w||^2 + b^2) + mean hinge.
    lin = models["clf::sklearn LinearSVC"][-1]
    ypm = np.where(D.tier[0] == 1, 1.0, -1.0)
    lam = svm.lambda_

    def primal(margin, sq_norm):
        return 0.5 * lam * sq_norm + float(np.mean(np.maximum(0.0, 1.0 - ypm * margin)))

    w_l, b_l = lin.coef_.ravel(), float(lin.intercept_[0])
    res["svm_vs_liblinear_objective"] = {
        "pegasos": primal(svm.decision_function(D.Z[0]), float(svm.w @ svm.w)),
        "liblinear": primal(D.Z[0] @ w_l + b_l, float(w_l @ w_l + b_l ** 2)),
        "prediction_agreement_test": float(np.mean(preds["clf::Ours: linear SVM (Pegasos)"][0]
                                                   == preds["clf::sklearn LinearSVC"][0])),
    }

    # --- error analysis (Task A, our final tree) ----------------------------------------
    pred = preds["reg::Ours: decision tree"]
    resid = pred - y_te
    absr = np.abs(resid)
    groups = {}
    cat = te_df["category"].fillna("(missing)")
    cats = cat.value_counts().index[:7]
    groups["By category"] = ([c for c in cats], [float(absr[(cat == c).to_numpy()].mean()) for c in cats],
                             [int((cat == c).sum()) for c in cats])
    # sparse districts: how many TRAINING listings share the district
    loc_counts = data.df.iloc[data.dev]["location"].value_counts()
    n_loc = te_df["location"].map(loc_counts).fillna(0).to_numpy()
    edges = [0, 20, 100, 500, np.inf]
    labels = ["<20", "20–99", "100–499", "500+"]
    b = np.digitize(n_loc, edges[1:-1])
    keep = [k for k in range(len(labels)) if np.any(b == k)]
    groups["By listings in district (dev)"] = ([labels[k] for k in keep],
                                               [float(absr[b == k].mean()) for k in keep],
                                               [int((b == k).sum()) for k in keep])
    dec = np.digitize(y_te, np.quantile(y_te, np.linspace(0, 1, 11)[1:-1]))
    groups["By true-price decile"] = ([f"D{k + 1}" for k in range(10)],
                                      [float(absr[dec == k].mean()) for k in range(10)],
                                      [int((dec == k).sum()) for k in range(10)])
    res["figures"].append(plots.error_by_group(groups, out))
    res["error_groups"] = {k: list(zip(v[0], v[1], v[2])) for k, v in groups.items()}
    dec_resid = {"Our tree": [float(resid[dec == k].mean()) for k in range(10)],
                 "Our ridge (bonus)": [float((preds["reg::Ours: ridge (bonus)"] - y_te)[dec == k].mean())
                                       for k in range(10)]}
    res["figures"].append(plots.residual_by_decile(np.arange(1, 11), dec_resid, out))
    res["decile_residuals"] = dec_resid
    res["figures"].append(plots.residual_map(te_df["lat"].to_numpy(), te_df["lng"].to_numpy(), resid, out))
    worst = np.argsort(absr)[::-1][:10]
    res["worst_predictions"] = [
        dict(true_azn=float(np.exp(y_te[i])), pred_azn=float(np.exp(pred[i])),
             area_m2=float(te_df["area_m2"].iloc[i]), rooms=float(te_df["rooms"].iloc[i]),
             category=str(te_df["category"].iloc[i]), location=str(te_df["location"].iloc[i]))
        for i in worst]
    res["share_abs_err_gt_50pct"] = float(np.mean(np.abs(np.exp(resid) - 1) > 0.5))

    # --- Task B: errors concentrate near the threshold ------------------------------------
    dist = np.abs(y_te - np.log(fin["threshold_dev"]))
    bins = [0, 0.05, 0.1, 0.2, 0.4, np.inf]
    blabels = ["<0.05", "0.05–0.1", "0.1–0.2", "0.2–0.4", ">0.4"]
    bi = np.digitize(dist, bins[1:-1])
    tier_pred = preds["clf::Ours: decision tree"][0]
    svm_pred = preds["clf::Ours: linear SVM (Pegasos)"][0]
    err = [float(np.mean(tier_pred[bi == k] != t_te[bi == k])) if np.any(bi == k) else 0.0
           for k in range(len(blabels))]
    cnt = [int(np.sum(bi == k)) for k in range(len(blabels))]
    res["figures"].append(plots.misclassification_vs_distance(blabels, err, cnt, out))
    near = dist < 0.1
    res["tier_errors"] = dict(bins=blabels, tree_error_rate=err, counts=cnt,
                              share_of_tree_errors_within_10pct=float(np.mean(near[tier_pred != t_te])),
                              share_of_svm_errors_within_10pct=float(np.mean(near[svm_pred != t_te])),
                              share_of_rows_within_10pct=float(np.mean(near)))
    return res


# =============================================================================
# 8. Bonus: ensembles from our own trees, k-means / PCA
# =============================================================================
def bonus_ensembles(data: Data, s: Settings, tree_res, fin) -> dict:
    D = fin["_design"]
    X0, X1 = D.X
    y0, y1 = D.ylog
    t0, t1 = D.tier
    out = {}
    times = {}
    with ev.timer(times, "rf_reg_fit_s"):
        rf = PerM2Target(RandomForest("regression", n_estimators=s.forest_trees, max_features=1 / 3,
                                      min_samples_leaf=config.FOREST_MIN_LEAF, random_state=SEED)).fit(X0, y0)
    rf_curve = [ev.rmse(y1, p) for p in rf.staged_output(X1)]
    with ev.timer(times, "sk_rf_reg_fit_s"):
        skrf = PerM2Target(SkRFR(n_estimators=s.forest_trees, max_features=1 / 3,
                                 min_samples_leaf=config.FOREST_MIN_LEAF, random_state=SEED,
                                 n_jobs=1)).fit(X0, y0)
    sk_rf_rmse = ev.rmse(y1, skrf.predict(X1))
    with ev.timer(times, "gb_fit_s"):
        gb = PerM2Target(GradientBoostingRegressor(n_estimators=s.boost_trees, learning_rate=config.BOOST_LR,
                                                   max_depth=config.BOOST_DEPTH, random_state=SEED)).fit(X0, y0)
    gb_curve = [ev.rmse(y1, p) for p in gb.staged_predict(X1)]
    with ev.timer(times, "sk_gb_fit_s"):
        skgb = PerM2Target(SkGB(n_estimators=s.boost_trees, learning_rate=config.BOOST_LR,
                                max_depth=config.BOOST_DEPTH, random_state=SEED)).fit(X0, y0)
    sk_gb_rmse = ev.rmse(y1, skgb.predict(X1))
    with ev.timer(times, "rf_clf_fit_s"):
        rfc = RandomForest("classification", n_estimators=s.forest_trees, max_features="sqrt",
                           criterion=tree_res["best_clf"]["criterion"], random_state=SEED).fit(X0, t0)
    proba = rfc.predict_proba(X1)[:, 1]
    rfc_pred = (proba > 0.5).astype(int)
    skrfc = SkRFC(n_estimators=s.forest_trees, max_features="sqrt",
                  criterion=tree_res["best_clf"]["criterion"], random_state=SEED, n_jobs=1).fit(X0, t0)
    single = fin["regression"]["Ours: decision tree"]["metrics"]["rmse_log"]
    fig = plots.ensemble_curves(np.arange(1, len(rf_curve) + 1), rf_curve, sk_rf_rmse,
                                np.arange(1, len(gb_curve) + 1), gb_curve, sk_gb_rmse,
                                (single,), s.figures_dir)
    out.update(
        rf_reg={"ours": ev.regression_metrics(y1, rf.predict(X1)), "sklearn": ev.regression_metrics(y1, skrf.predict(X1))},
        gb_reg={"ours": ev.regression_metrics(y1, gb.predict(X1)), "sklearn": ev.regression_metrics(y1, skgb.predict(X1))},
        rf_clf={"ours": ev.classification_metrics(t1, rfc_pred, proba),
                "sklearn": ev.classification_metrics(t1, skrfc.predict(X1), skrfc.predict_proba(X1)[:, 1])},
        curves={"rf": rf_curve, "gb": gb_curve}, times=times, figures=[fig],
        rf_importance_top=[(D.names[i], float(rf.feature_importances_[i]))
                           for i in np.argsort(rf.feature_importances_)[::-1][:8]],
        _models={"Our random forest": ("X", rf), "Our gradient boosting": ("X", gb)},
    )
    ens_preds = {"reg::Ours: random forest": rf.predict(X1), "reg::sklearn RandomForest": skrf.predict(X1),
                 "reg::Ours: gradient boosting": gb.predict(X1), "reg::sklearn GradientBoosting": skgb.predict(X1),
                 "reg::Ours: decision tree": fin["_predictions"]["reg::Ours: decision tree"],
                 "clf::Ours: random forest": (rfc_pred, proba),
                 "clf::sklearn RandomForest": (skrfc.predict(X1), skrfc.predict_proba(X1)[:, 1]),
                 "clf::Ours: decision tree": fin["_predictions"]["clf::Ours: decision tree"]}
    out["paired"] = paired_tests(y1, t1, ens_preds, s.n_boot, config.ENSEMBLE_PAIRS)
    log_step(f"  RF RMSE(log) ours={rf_curve[-1]:.4f} sklearn={sk_rf_rmse:.4f}; "
             f"GB ours={gb_curve[-1]:.4f} sklearn={sk_gb_rmse:.4f}")
    return out


def bonus_unsupervised(data: Data, s: Settings, fin, tree_res) -> dict:
    """k-means on dev coordinates -> neighbourhood clusters for error analysis; PCA."""
    dev_df = data.df.iloc[data.dev]
    te_df = data.df.iloc[data.te]
    ok = dev_df[["lat", "lng"]].notna().all(axis=1).to_numpy()
    xy = latlng_to_km(dev_df["lat"].to_numpy()[ok], dev_df["lng"].to_numpy()[ok])
    ks = list(range(2, 21)) if not s.fast else [2, 4, 8, 12]
    inertia = [KMeans(k, n_init=3, random_state=SEED).fit(xy).inertia_ for k in ks]
    km = KMeans(config.KMEANS_K, n_init=5, random_state=SEED).fit(xy)
    logp_dev = np.log(data.price[data.dev][ok])
    cl_med = np.array([np.median(logp_dev[km.labels_ == k]) if np.any(km.labels_ == k) else np.nan
                       for k in range(config.KMEANS_K)])
    fig1 = _kmeans_fig(ks, inertia, xy, km, cl_med, s)

    # cluster-level error analysis on test (our tree)
    D = fin["_design"]
    resid = fin["_predictions"]["reg::Ours: decision tree"] - D.ylog[1]
    ok_te = te_df[["lat", "lng"]].notna().all(axis=1).to_numpy()
    lab = np.full(len(te_df), -1)
    lab[ok_te] = km.predict(latlng_to_km(te_df["lat"].to_numpy()[ok_te], te_df["lng"].to_numpy()[ok_te]))
    per = []
    for k in range(config.KMEANS_K):
        m = lab == k
        if m.sum():
            per.append(dict(cluster=k, n_test=int(m.sum()), n_dev=int((km.labels_ == k).sum()),
                            median_price=float(np.exp(cl_med[k])),
                            mae_log=float(np.abs(resid[m]).mean()), bias_log=float(resid[m].mean())))

    # ablation on VALIDATION: does adding cluster one-hots help the tree?
    tr_df, va_df = data.df.iloc[data.tr], data.df.iloc[data.va]
    Dv = design(data, data.tr, data.va)
    km_tr_ok = tr_df[["lat", "lng"]].notna().all(axis=1).to_numpy()
    km_tr = KMeans(config.KMEANS_K, n_init=5, random_state=SEED).fit(
        latlng_to_km(tr_df["lat"].to_numpy()[km_tr_ok], tr_df["lng"].to_numpy()[km_tr_ok]))

    def onehot(frame):
        okf = frame[["lat", "lng"]].notna().all(axis=1).to_numpy()
        labf = np.full(len(frame), -1)
        labf[okf] = km_tr.predict(latlng_to_km(frame["lat"].to_numpy()[okf], frame["lng"].to_numpy()[okf]))
        return np.eye(config.KMEANS_K + 1)[labf + 1][:, 1:]

    best = tree_res["best_reg"]
    kw = dict(max_depth=best["max_depth"], min_samples_leaf=best["min_samples_leaf"],
              min_impurity_decrease=best["min_impurity_decrease"])
    base = PerM2Target(DecisionTree("regression", **kw)).fit(Dv.X[0], Dv.ylog[0])
    aug = PerM2Target(DecisionTree("regression", **kw)).fit(np.hstack([Dv.X[0], onehot(tr_df)]), Dv.ylog[0])
    ablation = dict(val_rmse_without=ev.rmse(Dv.ylog[1], base.predict(Dv.X[1])),
                    val_rmse_with_clusters=ev.rmse(Dv.ylog[1], aug.predict(np.hstack([Dv.X[1], onehot(va_df)]))))

    pca = PCA().fit(Dv.Z[0])
    Zp = pca.transform(Dv.Z[1])[:, :2]
    fig2 = plots.pca_figure(pca.explained_variance_ratio_, Zp, Dv.tier[1], s.figures_dir)
    evr = np.cumsum(pca.explained_variance_ratio_)
    return dict(inertia={"k": ks, "inertia": inertia}, clusters=per, ablation=ablation,
                pca_components_for_90pct=int(np.searchsorted(evr, 0.9) + 1),
                pca_pc1_pc2_share=float(evr[1]), figures=[fig1, fig2])


def bonus_svr(data: Data, s: Settings, fin) -> dict:
    """
    Bonus (Problem 8): our SVM turned into a Task A regressor (src/svr.py).
    lambda, epsilon (and the RFF bandwidth) are chosen on VALIDATION; the
    chosen configs are refitted on dev and scored on the test split once,
    next to sklearn's LinearSVR / SVR(rbf) with matched C = 1/(n lambda).
    """
    lambdas = config.FAST["svr_lambdas"] if s.fast else config.SVR_LAMBDAS
    gammas = config.FAST["svr_rff_gammas"] if s.fast else config.SVR_RFF_GAMMAS
    epochs = s.svm_epochs if s.fast else config.SVR_EPOCHS
    kw = dict(n_epochs=epochs, batch_size=s.svm_batch, random_state=SEED, record_objective=False)
    Dv = design(data, data.tr, data.va)
    grid = []
    for eps in config.SVR_EPSILONS:
        for lam in lambdas:
            m = PegasosSVR(lambda_=lam, epsilon=eps, **kw).fit(Dv.Z[0], Dv.ylog[0])
            grid.append(dict(lambda_=lam, epsilon=eps, val_rmse=ev.rmse(Dv.ylog[1], m.predict(Dv.Z[1]))))
    best = min(grid, key=lambda r: r["val_rmse"])
    rff_grid = []
    for g in gammas:
        for lam in config.SVR_RFF_LAMBDAS if not s.fast else lambdas:
            m = RFFPegasosSVR(gamma=g, n_components=s.rff_components, lambda_=lam,
                              epsilon=best["epsilon"], **kw).fit(Dv.Z[0], Dv.ylog[0])
            rff_grid.append(dict(gamma=g, lambda_=lam, val_rmse=ev.rmse(Dv.ylog[1], m.predict(Dv.Z[1]))))
    best_rff = min(rff_grid, key=lambda r: r["val_rmse"])
    log_step(f"  SVR selected: linear lambda={best['lambda_']:g} eps={best['epsilon']} "
             f"(val {best['val_rmse']:.4f}); RFF gamma={best_rff['gamma']} lambda={best_rff['lambda_']:g} "
             f"(val {best_rff['val_rmse']:.4f})")

    D = fin["_design"]
    n = len(D.ylog[0])
    eps = best["epsilon"]
    models = {
        "Ours: linear SVR": lambda: PegasosSVR(lambda_=best["lambda_"], epsilon=eps,
                                               **{**kw, "record_objective": True}),
        "sklearn LinearSVR": lambda: LinearSVR(C=1.0 / (n * best["lambda_"]), epsilon=eps,
                                               loss="epsilon_insensitive", dual=True,
                                               max_iter=5_000, random_state=SEED),
        "Ours: RBF SVR (RFF)": lambda: RFFPegasosSVR(gamma=best_rff["gamma"], n_components=s.rff_components,
                                                     lambda_=best_rff["lambda_"], epsilon=eps, **kw),
        "sklearn SVR (RBF)": lambda: SVR(kernel="rbf", gamma=best_rff["gamma"],
                                         C=1.0 / (n * best_rff["lambda_"]), epsilon=eps),
    }
    res, fitted = {}, {}
    rng_idx = np.random.default_rng(SEED).choice(n, min(n, s.svc_max_train), replace=False)
    for name, make in models.items():
        idx = rng_idx if name == "sklearn SVR (RBF)" else slice(None)   # O(n^2) kernel solver
        model, times = make(), {}
        with ev.timer(times, "fit_s"):
            model.fit(D.Z[0][idx], D.ylog[0][idx])
        with ev.timer(times, "predict_s"):
            pred = model.predict(D.Z[1])
        res[name] = {"metrics": ev.regression_metrics(D.ylog[1], pred), "times": times}
        fitted[name] = model
        log_step(f"  [A] {name:34s} RMSE(log)={res[name]['metrics']['rmse_log']:.4f} fit={times['fit_s']:.2f}s")

    # same primal for both linear solvers -> compare the objective each one reaches
    ours, sk = fitted["Ours: linear SVR"], fitted["sklearn LinearSVR"]
    Xa, yc = ours._augment(D.Z[0]), D.ylog[0] - ours.y_mean_
    w_sk = np.r_[sk.coef_, sk.intercept_[0] - ours.y_mean_]
    obj = {"ours": ours.objective(Xa, yc)[0], "sklearn": ours.objective(Xa, yc, w_sk)[0],
           "history": ours.history_["objective"]}
    return dict(grid=grid, best=best, rff_grid=rff_grid, best_rff=best_rff, test=res, objective=obj,
                tube_fraction_test=ours.tube_fraction(D.Z[1], D.ylog[1]),
                svr_subsample=int(len(rng_idx)))


def _kmeans_fig(ks, inertia, xy, km, cl_med, s):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(plots.FULL_W, 2.4), layout="constrained")
    axes[0].plot(ks, inertia, marker="o", color=plots.SERIES[0])
    axes[0].axvline(config.KMEANS_K, color=plots.MUTED, linewidth=0.7)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (km²)")
    axes[0].set_title("(a) Elbow of k-means inertia", loc="left", fontweight="bold")
    vals = cl_med[km.labels_]
    box = latlng_to_km(np.array(config.BAKU_BBOX["lat"]), np.array(config.BAKU_BBOX["lng"]))
    sc = axes[1].scatter(xy[:, 0], xy[:, 1], c=vals, cmap=plots.SEQ, s=1, linewidths=0, rasterized=True)
    for k, (cx, cy) in enumerate(km.cluster_centers_):
        if not (box[0, 0] < cx < box[1, 0] and box[0, 1] < cy < box[1, 1]):
            continue
        axes[1].text(cx, cy, str(k), ha="center", va="center", fontsize=6, color=plots.INK,
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=plots.AXIS, lw=0.4))
    axes[1].set_xlim(box[0, 0], box[1, 0])
    axes[1].set_ylim(box[0, 1], box[1, 1])
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("km east")
    axes[1].set_ylabel("km north")
    axes[1].set_title(f"(b) k={config.KMEANS_K} clusters (Baku zoom)", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=axes[1], shrink=0.85)
    cb.set_label("Cluster median log(price)")
    cb.outline.set_visible(False)
    return plots._save(fig, s.figures_dir, "bonus_kmeans")
