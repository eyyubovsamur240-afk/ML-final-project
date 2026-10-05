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
    for table in ("tab_results_reg", "tab_results_clf", "tab_headline", "tab_cleaning", "tab_tree_compare"):
        assert (out / "generated" / f"{table}.tex").exists()
    figures = {p.stem for p in (out / "figures").glob("*.pdf")}
    assert {"tree_depth_curves", "svm_convergence", "svm_lambda_sweep", "results_roc_pr",
            "feature_importance", "error_by_group"} <= figures
    # our tree and sklearn's never pick splits with different gains
    assert R["analysis"]["tree_vs_sklearn"]["regression"]["structure"]["gain_mismatch"] == 0
    # the test split was never used for selection: selections are recorded from validation
    assert "val_rmse" in R["tree"]["best_reg"] and "val_f1" in R["svm"]["best"]
