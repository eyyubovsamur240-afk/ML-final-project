"""End-to-end smoke test: run_all on synthetic data writes every artefact."""

import json

import pytest

from src import run_all
from tests.fake_binaaz import make_fake_binaaz


@pytest.mark.slow
def test_run_all_fast_end_to_end(tmp_path):
    csv = tmp_path / "fake.csv"
    make_fake_binaaz(2500, seed=3).to_csv(csv, index=False)
    out = tmp_path / "out"
    R = run_all.main(["--fast", "--data", str(csv), "--out-dir", str(out)])

    metrics = json.loads((out / "results" / "metrics.json").read_text())
    assert metrics["data"]["n_clean"] > 1000
    assert {"unit_price", "total_price"} <= set(metrics["data"]["dropped_columns"]["leakage"])
    assert (out / "generated" / "macros.tex").exists()
    for table in ("tab_results_reg", "tab_results_clf", "tab_headline", "tab_cleaning", "tab_tree_compare",
                  "tab_error_district", "tab_error_band", "tab_paired"):
        assert (out / "generated" / f"{table}.tex").exists()
    figures = {p.stem for p in (out / "figures").glob("*.pdf")}
    assert {"tree_depth_curves", "svm_convergence", "svm_lambda_sweep", "results_roc_pr",
            "feature_importance", "error_by_group", "feature_importance_permutation",
            "error_breakdown"} <= figures
    # explanations cover the forest too, and every price band adds up to the test split
    E = metrics["explain"]
    assert "Our random forest" in E["perm_importance_reg"]
    assert sum(r["n"] for r in E["by_price_band"]) == metrics["data"]["n_test"]
    assert sum(r["n"] for r in E["by_district"]) == metrics["data"]["n_test"]
    # our tree and sklearn's never pick splits with different gains
    assert R["analysis"]["tree_vs_sklearn"]["regression"]["structure"]["gain_mismatch"] == 0
    # the test split was never used for selection: selections are recorded from validation
    assert "val_rmse" in R["tree"]["best_reg"] and "val_f1" in R["svm"]["best"]
    # every planned paired comparison ran, each with a CI around its point difference
    paired = metrics["final"]["paired"] + metrics["bonus"]["ensembles"]["paired"]
    assert len(paired) == 3 + 5 * 2 + 3 + 2 * 2       # Task A: RMSE; Task B: F1 + AUC
    assert all(r["ci"][0] <= r["diff"] <= r["ci"][1] for r in paired)
    assert (out / "results" / "paired_tests.csv").exists()
