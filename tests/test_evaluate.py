"""Our NumPy metrics must agree with scikit-learn's."""

import numpy as np
import pytest
from sklearn import metrics as skm

from src import evaluate as ev


@pytest.fixture
def reg():
    rng = np.random.default_rng(0)
    y = rng.normal(10, 2, 300)
    return y, y + rng.normal(0, 1, 300)


@pytest.fixture
def clf():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    scores = np.round(y + rng.normal(0, 1, 400), 1)          # rounded -> many ties
    return y, (scores > 0.5).astype(int), scores


def test_regression_metrics(reg):
    y, p = reg
    assert ev.rmse(y, p) == pytest.approx(np.sqrt(skm.mean_squared_error(y, p)))
    assert ev.mae(y, p) == pytest.approx(skm.mean_absolute_error(y, p))
    assert ev.r2(y, p) == pytest.approx(skm.r2_score(y, p))
    assert ev.mape(y, p) == pytest.approx(skm.mean_absolute_percentage_error(y, p))


def test_confusion_and_prf(clf):
    y, yhat, _ = clf
    np.testing.assert_array_equal(ev.confusion_matrix(y, yhat), skm.confusion_matrix(y, yhat))
    p, r, f1 = ev.precision_recall_f1(y, yhat)
    assert p == pytest.approx(skm.precision_score(y, yhat))
    assert r == pytest.approx(skm.recall_score(y, yhat))
    assert f1 == pytest.approx(skm.f1_score(y, yhat))
    assert ev.accuracy(y, yhat) == pytest.approx(skm.accuracy_score(y, yhat))


def test_auc_and_average_precision_with_ties(clf):
    y, _, s = clf
    assert ev.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert ev.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))
    fpr, tpr, _ = ev.roc_curve(y, s)
    fpr2, tpr2, _ = skm.roc_curve(y, s, drop_intermediate=False)
    np.testing.assert_allclose(fpr, fpr2)
    np.testing.assert_allclose(tpr, tpr2)


def test_rankdata_average_ties():
    np.testing.assert_allclose(ev.rankdata([10, 20, 20, 5]), [2, 3.5, 3.5, 1])


def test_degenerate_cases():
    assert ev.precision_recall_f1([0, 0], [0, 0]) == (0.0, 0.0, 0.0)
    assert np.isnan(ev.roc_auc([1, 1], [0.2, 0.3]))


def test_bootstrap_ci_contains_point_estimate(reg):
    y, p = reg
    lo, hi = ev.bootstrap_ci(ev.rmse, y, p, n_boot=300)
    assert lo < ev.rmse(y, p) < hi


def test_compare_prints_table(capsys):
    table = ev.compare({"ours": {"f1": 0.8, "auc": (0.9, 0.01)}, "sklearn": {"f1": 0.81, "auc": (0.91, 0.02)}})
    assert "ours" in capsys.readouterr().out
    assert "0.9000 ± 0.0100" in table
