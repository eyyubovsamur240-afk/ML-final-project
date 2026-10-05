import numpy as np

from src import explain


def test_feature_groups_merge_onehots_and_indicators():
    names = ["area_m2", "log_area", "rooms", "rooms_missing", "district=a", "district=b", "district=other",
             "city=x"]
    g = explain.feature_groups(names)
    assert g == {"area": [0, 1], "rooms": [2, 3], "district": [4, 5, 6], "city": [7]}


def test_permutation_importance_finds_the_only_used_feature():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = 2 * X[:, 1]
    groups = {"a": [0], "b": [1], "c": [2]}
    imp = explain.permutation_importance(lambda M: float(np.sqrt(np.mean((2 * M["X"][:, 1] - y) ** 2))),
                                         {"X": X}, groups, n_repeats=2)
    assert imp["a"] == (0.0, 0.0) and imp["c"] == (0.0, 0.0)
    assert imp["b"][0] > 1.0


def test_permutation_shuffles_all_matrices_alike():
    X = np.arange(20, dtype=float).reshape(10, 2)
    Z = X * 10
    seen = []
    explain.permutation_importance(lambda M: seen.append((M["X"].copy(), M["Z"].copy())) or 0.0,
                                   {"X": X, "Z": Z}, {"g": [0]}, n_repeats=1)
    Xs, Zs = seen[-1]
    assert np.array_equal(Xs * 10, Zs)
    assert np.array_equal(Xs[:, 1], X[:, 1])


def test_price_bands_and_districts():
    idx, labels = explain.price_band_labels(np.array([50_000, 75_000, 1_000_000]))
    assert [labels[i] for i in idx] == ["<75k", "75k–125k", "600k+"]
    d = np.array(["a"] * 40 + ["b"] * 5 + [np.nan] * 3, dtype=object)
    lab, keep = explain.district_labels(d, max_n=5, min_rows=30)
    assert keep == ["a"]
    assert set(lab) == {"a", "other", "(none)"}
