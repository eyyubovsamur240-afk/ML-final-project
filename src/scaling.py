"""
scaling.py — how fit and predict time grow with the number of training rows
(Problem 5c: "a note on fit/predict time and scaling").

Every model is trained at its SELECTED configuration on nested random
subsamples of the training split (sizes from ``config.SCALING_FRACTIONS``)
and timed; predict time is measured on the full validation split. The
empirical exponent b in  time ~ a * n^b  is the slope of a least-squares line
through the log-log points (largest sizes only, where fixed overheads no
longer dominate). Theory for comparison:

  * CART (ours and sklearn): O(d n log n) per tree level -> b ~ 1.0-1.2;
  * Pegasos / SGD / LIBLINEAR: a fixed number of epochs, each O(n d) -> b ~ 1;
  * RFF + Pegasos: O(n D) per epoch -> b ~ 1, with a larger constant;
  * kernel SVC (libsvm, SMO): between O(n^2) and O(n^3) -> b ~ 2,
    which is why the main benchmark trains it on a subsample.

Only timing is measured here; no score from this stage is used to pick
anything.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config, plots
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RFFPegasosSVM

SEED = config.SEED


def _models(tree_res, svm_res, s):
    """(label, task, matrix, factory, max_n). matrix: 'X' raw, 'Z' standardised."""
    r, c = tree_res["best_reg"], tree_res["best_clf"]
    kw_r = dict(max_depth=r["max_depth"], min_samples_leaf=r["min_samples_leaf"],
                min_impurity_decrease=r["min_impurity_decrease"])
    kw_c = dict(max_depth=c["max_depth"], min_samples_leaf=c["min_samples_leaf"])
    lam = svm_res["best"]["lambda_"]
    rff = svm_res["best_rff"]
    svm_kw = dict(n_epochs=s.svm_epochs, batch_size=s.svm_batch, random_state=SEED,
                  average=s.svm_average, record_objective=False)
    return [
        ("Ours: tree (A)", "reg", "X", lambda n: DecisionTree("regression", "mse", random_state=SEED, **kw_r), None),
        ("sklearn tree (A)", "reg", "X", lambda n: DecisionTreeRegressor(random_state=SEED, **kw_r), None),
        ("Ours: tree (B)", "clf", "X",
         lambda n: DecisionTree("classification", c["criterion"], random_state=SEED, **kw_c), None),
        ("sklearn tree (B)", "clf", "X",
         lambda n: DecisionTreeClassifier(criterion=c["criterion"], random_state=SEED, **kw_c), None),
        ("Ours: Pegasos", "clf", "Z", lambda n: PegasosSVM(lambda_=lam, **svm_kw), None),
        ("sklearn SGDClassifier", "clf", "Z", lambda n: SGDClassifier(
            loss="hinge", alpha=lam, max_iter=s.svm_epochs, tol=None, random_state=SEED), None),
        ("sklearn LinearSVC", "clf", "Z", lambda n: LinearSVC(
            C=1.0 / (n * lam), loss="hinge", dual=True, max_iter=200_000, random_state=SEED), None),
        ("Ours: RFF + Pegasos", "clf", "Z", lambda n: RFFPegasosSVM(
            gamma=rff["gamma"], n_components=s.rff_components, lambda_=rff["lambda_"], **svm_kw), None),
        ("sklearn SVC (RBF)", "clf", "Z", lambda n: SVC(
            kernel="rbf", gamma=rff["gamma"], C=1.0 / (n * rff["lambda_"])), s.svc_max_train),
    ]


def loglog_slope(n, t, last: int = 3) -> float:
    """Slope of log(t) on log(n) through the ``last`` largest sizes."""
    n, t = np.asarray(n, float)[-last:], np.asarray(t, float)[-last:]
    ok = t > 0
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(n[ok]), np.log(t[ok]), 1)[0])


def scaling_study(D, s, tree_res, svm_res, fractions=None) -> dict:
    """
    ``D`` is a design fitted on the training split with the validation split
    as element 1 (``experiments.design(data, data.tr, data.va)``).
    """
    fractions = fractions or (config.FAST["scaling_fractions"] if s.fast else config.SCALING_FRACTIONS)
    n_tr = len(D.ylog[0])
    order = np.random.default_rng(SEED).permutation(n_tr)       # nested subsamples
    sizes = sorted({max(50, int(round(f * n_tr))) for f in fractions})
    out = {"sizes": sizes, "models": {}}
    for label, task, kind, make, max_n in _models(tree_res, svm_res, s):
        X0 = D.Z[0] if kind == "Z" else D.X[0]
        X1 = D.Z[1] if kind == "Z" else D.X[1]
        y0 = D.ylog[0] if task == "reg" else D.tier[0]
        rec = {"n": [], "fit_s": [], "predict_s": []}
        for n in sizes:
            if max_n and n > max_n:
                break
            idx = order[:n]
            model = make(n)
            t0 = time.perf_counter()
            model.fit(X0[idx], y0[idx])
            t1 = time.perf_counter()
            model.predict(X1)
            t2 = time.perf_counter()
            rec["n"].append(n)
            rec["fit_s"].append(t1 - t0)
            rec["predict_s"].append(t2 - t1)
        rec["fit_exponent"] = loglog_slope(rec["n"], rec["fit_s"])
        out["models"][label] = rec
    out["figures"] = [scaling_figure(out, s.figures_dir)]
    return out


# colour per model family (ours solid, sklearn dashed in the same colour)
_COLOUR = {"Ours: tree (A)": 0, "sklearn tree (A)": 0, "Ours: tree (B)": 1, "sklearn tree (B)": 1,
           "Ours: Pegasos": 0, "sklearn SGDClassifier": 1, "sklearn LinearSVC": 2,
           "Ours: RFF + Pegasos": 3, "sklearn SVC (RBF)": 3}


def scaling_figure(res, out_dir):
    """Log-log fit time; ours solid, sklearn dashed, same colour within a family."""
    fig, axes = plots._subplots(2, plots.FULL_W, 2.9)
    groups = (("(a) Trees", lambda k: "tree" in k), ("(b) SVMs", lambda k: "tree" not in k))
    for ax, (title, keep) in zip(axes, groups):
        for label, rec in res["models"].items():
            if not keep(label) or not rec["n"]:
                continue
            ours = label.startswith("Ours")
            fam = _COLOUR.get(label, 4)
            ax.plot(rec["n"], rec["fit_s"], marker="o" if ours else "s",
                    linestyle="-" if ours else "--", color=plots.SERIES[fam],
                    label=f"{label}  (b={rec['fit_exponent']:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training listings n")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        plots._panel_label(ax, title)
    axes[0].set_ylabel("Fit time (s)")
    return plots._save(fig, out_dir, "scaling_fit_time")
