"""Tests for the from-scratch decision tree (sklearn is used only as the reference)."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree, entropy_from_counts, gini_from_counts
from src.tree_compare import compare_tree_structure


def sk_structure(sk):
    """sklearn tree -> (feature per node with -1 for leaves, thresholds of internal nodes)."""
    t = sk.tree_
    feat = np.where(t.children_left == -1, -1, t.feature)
    return feat, t.threshold[feat >= 0]


def our_structure(tree):
    feat = tree.tree_feature_
    return feat, tree.tree_threshold_[feat >= 0]


# --------------------------------------------------------------------------- impurity
def test_impurity_values():
    assert gini_from_counts(np.array([2, 2]), 4) == pytest.approx(0.5)
    assert gini_from_counts(np.array([4, 0]), 4) == pytest.approx(0.0)
    assert entropy_from_counts(np.array([2, 2]), 4) == pytest.approx(1.0)
    assert entropy_from_counts(np.array([4, 0]), 4) == pytest.approx(0.0)
    assert entropy_from_counts(np.array([1, 1, 1, 1]), 4) == pytest.approx(2.0)
    reg = DecisionTree(task="regression")
    assert reg._impurity(np.array([1.0, 3.0])) == pytest.approx(1.0)   # variance


# --------------------------------------------------------------------------- exact match
TINY_X = np.array([[1, 1], [2, 3], [3, 4], [4, 5], [5, 2], [6, 7], [7, 6], [8, 8]], float)
TINY_Y = np.array([0, 0, 0, 1, 1, 1, 0, 0])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_matches_sklearn_on_tiny_determined_example(criterion):
    ours = DecisionTree(task="classification", criterion=criterion).fit(TINY_X, TINY_Y)
    sk = DecisionTreeClassifier(criterion=criterion, random_state=0).fit(TINY_X, TINY_Y)
    f1, t1 = our_structure(ours)
    f2, t2 = sk_structure(sk)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_allclose(t1, t2)
    # root split is x0 <= 3.5 (isolates the first three zeros)
    assert ours.root.feature == 0 and ours.root.threshold == pytest.approx(3.5)
    np.testing.assert_array_equal(ours.predict(TINY_X), TINY_Y)


def _int_data(n=300, d=5, seed=0, task="classification"):
    """Integer-valued, tie-free features (exact in float32, like sklearn uses)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.permutation(n) for _ in range(d)]).astype(float)
    signal = X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 40, n)
    if task == "classification":
        return X, (signal > np.median(signal)).astype(int)
    return X, signal


def assert_equivalent_up_to_ties(ours, sk, X, y):
    """Every node: same split decision and the same (optimal) gain as sklearn."""
    stats = compare_tree_structure(ours, sk, X, y)
    assert stats["gain_mismatch"] == 0, stats
    assert stats["decision_mismatch"] == 0, stats
    assert stats["max_threshold_diff"] < 1e-9, stats
    assert stats["same_split"] > 0, stats
    return stats


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("params", [
    {},
    {"max_depth": 3},
    {"max_depth": 4, "min_samples_leaf": 10},
    {"max_depth": 5, "min_samples_split": 40},
    {"min_impurity_decrease": 0.005},
])
def test_classifier_matches_sklearn_up_to_ties(criterion, params, seed):
    X, y = _int_data(seed=seed)
    ours = DecisionTree(task="classification", criterion=criterion, **params).fit(X, y)
    sk = DecisionTreeClassifier(criterion=criterion, random_state=0, **params).fit(X, y)
    stats = assert_equivalent_up_to_ties(ours, sk, X, y)
    if stats["tie_divergence"] == stats["same_partition"] == 0:   # no ties -> identical
        np.testing.assert_allclose(ours.feature_importances_, sk.feature_importances_,
                                   atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("params", [
    {},
    {"max_depth": 6},
    {"min_samples_leaf": 5},
    {"min_impurity_decrease": 5.0},
])
def test_regressor_matches_sklearn_up_to_ties(params, seed):
    X, y = _int_data(task="regression", seed=seed)
    ours = DecisionTree(task="regression", criterion="mse", **params).fit(X, y)
    sk = DecisionTreeRegressor(criterion="squared_error", random_state=0, **params).fit(X, y)
    stats = assert_equivalent_up_to_ties(ours, sk, X, y)
    if stats["tie_divergence"] == stats["same_partition"] == 0:
        Xt = np.random.default_rng(5).uniform(0, 300, size=(200, 5))
        np.testing.assert_allclose(ours.predict(Xt), sk.predict(Xt))


def test_regressor_structure_identical_without_ties():
    """Continuous target, nodes >= 10 rows: no ties occur, so the trees are identical."""
    X, y = _int_data(task="regression", seed=1)
    ours = DecisionTree(task="regression", max_depth=4, min_samples_leaf=10).fit(X, y)
    sk = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=0).fit(X, y)
    f1, t1 = our_structure(ours)
    f2, t2 = sk_structure(sk)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_allclose(t1, t2)
    np.testing.assert_allclose(ours.feature_importances_, sk.feature_importances_, atol=1e-10)


def test_full_tree_prediction_agreement_with_sklearn():
    """Fully grown trees can differ only through exact gain ties; agreement must be high."""
    X, y = _int_data(n=500, seed=3)
    ours = DecisionTree(task="classification").fit(X, y)
    sk = DecisionTreeClassifier(random_state=0).fit(X, y)
    Xt = np.random.default_rng(7).uniform(0, 500, size=(1000, 5))
    assert np.mean(ours.predict(Xt) == sk.predict(Xt)) > 0.9
    np.testing.assert_array_equal(ours.predict(X), y)      # pure leaves memorise train


# --------------------------------------------------------------------------- hyperparameters
def test_each_hyperparameter_changes_the_tree():
    X, y = _int_data(n=400, seed=2)
    base = DecisionTree(task="classification").fit(X, y)
    depth = DecisionTree(task="classification", max_depth=3).fit(X, y)
    split = DecisionTree(task="classification", min_samples_split=50).fit(X, y)
    leaf = DecisionTree(task="classification", min_samples_leaf=20).fit(X, y)
    dec = DecisionTree(task="classification", min_impurity_decrease=0.01).fit(X, y)
    assert depth.get_depth() == 3 < base.get_depth()
    for t in (split, leaf, dec):
        assert t.node_count < base.node_count
    leaves = leaf.tree_feature_ < 0
    assert leaf.tree_n_samples_[leaves].min() >= 20
    internal = split.tree_feature_ >= 0
    assert split.tree_n_samples_[internal].min() >= 50


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_truncated_tree_equals_refit_with_max_depth(task):
    X, y = _int_data(n=400, seed=4, task=task)
    full = DecisionTree(task=task, min_samples_leaf=3).fit(X, y)
    Xt = np.random.default_rng(1).uniform(0, 400, size=(300, 5))
    for d in (1, 2, 4, 7):
        refit = DecisionTree(task=task, min_samples_leaf=3, max_depth=d).fit(X, y)
        np.testing.assert_array_equal(full.predict(Xt, truncate_depth=d), refit.predict(Xt))


# --------------------------------------------------------------------------- API
def test_predict_proba_and_reference_traversal():
    X, y = _int_data(n=200, seed=6)
    tree = DecisionTree(task="classification", max_depth=4).fit(X, y)
    proba = tree.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(tree.predict(X), tree.classes_[proba.argmax(1)])
    slow = np.array([tree._predict_one(x, tree.root) for x in X])
    np.testing.assert_allclose(slow, proba)


def test_arbitrary_labels_and_constant_target():
    X = TINY_X
    labels = np.where(TINY_Y == 1, "premium", "standard")
    tree = DecisionTree(task="classification").fit(X, labels)
    assert set(tree.predict(X)) <= {"premium", "standard"}
    const = DecisionTree(task="regression").fit(X, np.ones(len(X)))
    assert const.node_count == 1 and const.predict(X[:2]).tolist() == [1.0, 1.0]


def test_invalid_criterion_raises():
    with pytest.raises(ValueError):
        DecisionTree(task="regression", criterion="gini")


def test_max_features_subsampling_is_seeded():
    X, y = _int_data(n=300, seed=8)
    a = DecisionTree(task="classification", max_features="sqrt", random_state=1).fit(X, y)
    b = DecisionTree(task="classification", max_features="sqrt", random_state=1).fit(X, y)
    np.testing.assert_array_equal(a.tree_feature_, b.tree_feature_)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_matches_sklearn_on_float_data_near_float32_resolution(task):
    """Latitude-like values: neighbouring float32 numbers are adjacent, the hardest case
    for threshold comparisons (sklearn stores float32 but compares in double)."""
    rng = np.random.default_rng(11)
    n = 3000
    X = np.column_stack([40.0 + rng.uniform(0, 0.01, n), 49.8 + rng.uniform(0, 0.01, n),
                         rng.normal(size=n)])
    signal = 300 * (X[:, 0] - 40.005) - 200 * (X[:, 1] - 49.805) + 0.3 * X[:, 2] + rng.normal(0, 0.5, n)
    y = signal if task == "regression" else (signal > 0).astype(int)
    kw = dict(max_depth=10, min_samples_leaf=10)
    ours = DecisionTree(task=task, **kw).fit(X, y)
    Sk = DecisionTreeRegressor if task == "regression" else DecisionTreeClassifier
    sk = Sk(random_state=0, **kw).fit(X, y)
    stats = compare_tree_structure(ours, sk, X, y)
    # every difference is explained by ties or by sklearn's float32 cast
    assert stats["gain_mismatch"] == 0 and stats["decision_mismatch"] == 0, stats
    assert stats["max_threshold_diff"] < 1e-5


def test_node_split_only_in_float64_is_a_float32_divergence():
    """Two rows whose feature differs only below float32 resolution: we split, sklearn cannot."""
    X = np.array([[40.38784118730932], [40.38784118730935], [41.0], [41.5]])
    y = np.array([0, 1, 1, 1])
    ours = DecisionTree(task="classification").fit(X, y)
    sk = DecisionTreeClassifier(random_state=0).fit(X, y)
    stats = compare_tree_structure(ours, sk, X, y)
    assert stats["decision_mismatch"] == 0 and stats["gain_mismatch"] == 0, stats
    assert stats["float32_divergence"] >= 1
