"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all                # full run (all figures, tables, CV)
    python -m src.run_all --fast         # small smoke test (~1 min)
    python -m src.run_all --data path/to/file.csv

Stages (see src/experiments.py for the protocol):
  1. load + clean the data, build features, stratified train/val/test split
  2. EDA figures (training split only)
  3. decision-tree studies + selection on validation   (Task A and Task B)
  4. Pegasos SVM studies + selection on validation     (Task B, linear + RFF)
  5. refit chosen configs on train+val, score the TEST split once
  6. k-fold CV on train+val for mean ± std (all models, sklearn baselines too)
  7. benchmark details + error analysis
  8. bonuses: ridge, random forest / gradient boosting from our trees, k-means, PCA
     then permutation importance per model + error breakdown by district / price band
  9. write results/ (json, csv) and report/generated/ (LaTeX macros + tables)

Everything is deterministic (config.SEED everywhere).
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from . import config, plots
from . import data_prep as dp
from . import evaluate as ev
from . import experiments as ex
from . import explain, reporting


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reproduce every number and figure of the report.")
    p.add_argument("--data", default=None, help="path to the bina.az CSV (default: data/)")
    p.add_argument("--fast", action="store_true", help="small, quick smoke-test run")
    p.add_argument("--skip-cv", action="store_true", help="skip k-fold CV (no ± std in tables)")
    p.add_argument("--skip-bonus", action="store_true", help="skip the bonus experiments")
    p.add_argument("--out-dir", default=None,
                   help="write results/, figures/ and generated/ here instead of the repo defaults")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    t_start = time.perf_counter()
    np.random.seed(config.SEED)        # belt and braces: all our code uses explicit RNGs
    if args.out_dir:
        base = Path(args.out_dir)
        results_dir, figures_dir, tex_dir = base / "results", base / "figures", base / "generated"
    else:
        results_dir, figures_dir, tex_dir = config.RESULTS_DIR, config.FIGURES_DIR, config.GENERATED_TEX_DIR
    s = ex.Settings.make(args.fast, figures_dir)
    plots.setup_style()
    R: dict = {"settings": {"fast": args.fast, "seed": config.SEED, "svm_epochs": s.svm_epochs,
                            "svm_batch": s.svm_batch, "rff_components": s.rff_components,
                            "forest_trees": s.forest_trees, "boost_trees": s.boost_trees,
                            "kmeans_k": config.KMEANS_K, "cv_folds": s.cv_folds}}

    # 1. data ----------------------------------------------------------------
    path = dp.find_data_file(args.data)
    ex.log_step(f"Loading {path}")
    data = ex.prepare_data(path, fast=args.fast)
    pre = dp.Preprocessor().fit(data.feats.iloc[data.tr])
    tier_tr, thr = dp.make_tier_label(data.price[data.tr])
    R["data"] = {
        "file": path.name, "sha256": file_sha256(path),
        "n_raw": data.log.steps[0][1], "n_clean": len(data.df),
        "n_train": len(data.tr), "n_val": len(data.va), "n_test": len(data.te),
        "n_features": len(pre.feature_names_), "n_numeric_features": len(pre.numeric) + len(pre.binary),
        "feature_names": pre.feature_names_,
        "cleaning_steps": data.log.steps, "dropped_columns": data.log.dropped_columns,
        "column_mapping": data.log.notes.get("column_mapping", {}),
        "currency_counts": data.log.notes.get("currency_counts", {}),
        "missing_after_cleaning": data.log.notes.get("missing_after_cleaning", {}),
        "scrape_period": data.log.notes.get("scrape_period"),
        "category_counts_before_scope": data.log.notes.get("category_counts_before_scope", {}),
        "tier_threshold_train": thr,
        "premium_share": {"train": float(tier_tr.mean()),
                          "val": float(dp.make_tier_label(data.price[data.va], thr)[0].mean()),
                          "test": float(dp.make_tier_label(data.price[data.te], thr)[0].mean())},
    }
    ex.log_step(f"{R['data']['n_raw']:,} raw rows -> {len(data.df):,} clean; split "
                f"{len(data.tr):,}/{len(data.va):,}/{len(data.te):,}; "
                f"{R['data']['n_features']} features; tier threshold {thr:,.0f} AZN")

    # 2-4. EDA and model studies ------------------------------------------------
    ex.log_step("EDA figures")
    R["eda"] = ex.eda(data, s)
    ex.log_step("Decision-tree studies (train -> validation)")
    R["tree"] = ex.tree_studies(data, s)
    ex.log_step("SVM studies (train -> validation)")
    R["svm"] = ex.svm_studies(data, s)
    alpha, ridge_rows = ex.select_ridge_alpha(data)
    R["ridge"] = {"alpha": alpha, "grid": ridge_rows}

    # 5. final models on the test split ---------------------------------------------
    ex.log_step("Final models: refit on train+val, evaluate on TEST")
    fin = ex.final_models(data, s, R["tree"], R["svm"], alpha)
    R["final"] = fin

    # 6. cross-validation ------------------------------------------------------------------
    if not args.skip_cv:
        ex.log_step(f"{s.cv_folds}-fold CV on train+val")
        R["cv"] = ex.cross_validate(data, s, R["tree"], R["svm"], alpha)

    # 7. analysis -----------------------------------------------------------------------------
    ex.log_step("Benchmark details and error analysis")
    R["analysis"] = ex.analysis(data, s, fin)

    # 8. bonus ----------------------------------------------------------------------------------
    if not args.skip_bonus:
        ex.log_step("Bonus: ensembles from our trees")
        ens = ex.bonus_ensembles(data, s, R["tree"], fin)
        ex.log_step("Bonus: k-means + PCA")
        uns = ex.bonus_unsupervised(data, s, fin, R["tree"])
        R["bonus"] = {"ensembles": ens, "unsupervised": uns}

    # 8b. explanations -------------------------------------------------------------------------
    ex.log_step("Feature importance (permutation) and error breakdown by district / price band")
    extra = R["bonus"]["ensembles"]["_models"] if "bonus" in R else None
    R["explain"] = explain.run(data, s, fin, extra, n_repeats=s.perm_repeats)

    # 9. outputs -----------------------------------------------------------------------------
    R["runtime_s"] = time.perf_counter() - t_start
    reporting.write_all(R, results_dir, tex_dir)
    _print_summary(R)
    ex.log_step(f"Done in {R['runtime_s'] / 60:.1f} min. Figures -> {figures_dir}, "
                f"tables/macros -> {tex_dir}, metrics -> {results_dir / 'metrics.json'}")
    return R


def _print_summary(R: dict) -> None:
    fin, cv = R["final"], R.get("cv")
    print()
    reg = {}
    for n, d in fin["regression"].items():
        m = d["metrics"]
        row = {"test RMSE(log)": m["rmse_log"], "test R2(log)": m["r2_log"],
               "test MAE(AZN)": round(m["mae_azn"]), "fit s": d["times"]["fit_s"]}
        if cv and n in cv["regression"]:
            row["CV RMSE(log)"] = cv["regression"][n]["rmse_log"]
        reg[n] = row
    ev.compare(reg, title="TASK A — regression of log(price)   [test split; CV = mean ± std on train+val]")
    print()
    clf = {}
    for n, d in fin["classification"].items():
        m = d["metrics"]
        row = {"test F1": m["f1"], "test ROC-AUC": m["roc_auc"], "test acc": m["accuracy"],
               "fit s": d["times"]["fit_s"]}
        if cv and n in cv["classification"]:
            row["CV F1"] = cv["classification"][n]["f1"]
        clf[n] = row
    ev.compare(clf, title="TASK B — price tier (premium vs standard)")
    a = R["analysis"]["tree_vs_sklearn"]
    print(f"\nTree vs sklearn — test agreement: regression {a['regression']['test_agreement']:.1%}, "
          f"classification {a['classification']['test_agreement']:.1%}; gain mismatches: "
          f"{a['regression']['structure']['gain_mismatch']} / {a['classification']['structure']['gain_mismatch']}")
    sm = R["analysis"]["svm_margin"]
    print(f"Pegasos SVM: ||w|| = {sm['w_norm']:.3f}, margin 2/||w|| = {sm['margin']:.3f}, "
          f"support vectors = {sm['n_sv']:,} ({sm['sv_frac']:.1%} of train)")


if __name__ == "__main__":
    main()
