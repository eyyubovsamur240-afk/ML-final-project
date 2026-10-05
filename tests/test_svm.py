"""Tests for the from-scratch Pegasos SVM and the random-Fourier-feature extension."""

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from src.svm import PegasosSVM, RandomFourierFeatures, RFFPegasosSVM, rbf_kernel


def blobs(n=200, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-sep / 2, 1, (n // 2, 2)), rng.normal(sep / 2, 1, (n // 2, 2))])
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    return X, y


def test_separable_data_is_classified_perfectly():
    X, y = blobs(sep=8.0)
    svm = PegasosSVM(lambda_=1e-3, n_epochs=20, batch_size=1).fit(X, y)
    assert np.mean(svm.predict(X) == y) == 1.0


@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("batch_size", [1, 16, None])
def test_objective_decreases(batch_size, average):
    X, y = blobs(sep=2.0)
    svm = PegasosSVM(lambda_=1e-2, n_epochs=30, batch_size=batch_size, average=average).fit(X, y)
    J = np.array(svm.history_["objective"])
    assert J[-1] < J[0]                     # J(w=0) = 1
    assert J[-1] <= np.min(J[1:]) + 0.02    # converged to (near) its best value


@pytest.mark.parametrize("batch_size, average", [(None, False), (16, True)])
def test_reaches_the_same_objective_as_liblinear(batch_size, average):
    """LinearSVC(C=1/(n*lambda)) with a regularised intercept solves the same primal."""
    X, y = blobs(n=400, sep=2.0, seed=3)
    lam = 1e-2
    ours = PegasosSVM(lambda_=lam, n_epochs=200, batch_size=batch_size, average=average).fit(X, y)
    sk = LinearSVC(C=1 / (len(y) * lam), loss="hinge", dual=True, max_iter=100_000,
                   tol=1e-8).fit(X, y)
    ypm = np.where(y == 1, 1.0, -1.0)

    def J(w, b):
        return 0.5 * lam * (w @ w + b * b) + np.mean(np.maximum(0, 1 - ypm * (X @ w + b)))

    j_ours = J(ours.coef_, ours.intercept_)
    j_sk = J(sk.coef_.ravel(), sk.intercept_[0])
    assert j_ours <= j_sk * 1.01
    assert np.mean(ours.predict(X) == sk.predict(X)) > 0.97


def test_lambda_controls_the_margin():
    X, y = blobs(sep=2.0)
    small = PegasosSVM(lambda_=1e-4, n_epochs=30, batch_size=8).fit(X, y)
    large = PegasosSVM(lambda_=1e-1, n_epochs=30, batch_size=8).fit(X, y)
    assert large.w_norm < small.w_norm
    assert large.margin_width > small.margin_width
    assert len(large.support_vectors(X, y)) > len(small.support_vectors(X, y))


def test_support_vectors_are_exactly_the_points_inside_the_margin():
    X, y = blobs(sep=2.0)
    svm = PegasosSVM(lambda_=1e-2, n_epochs=20, batch_size=8).fit(X, y)
    m = svm.functional_margins(X, y)
    sv = svm.support_vectors(X, y)
    assert np.all(m[sv] <= 1 + 1e-9)
    assert np.all(np.delete(m, sv) > 1)
    assert svm.margin_width == pytest.approx(2 / np.linalg.norm(svm.coef_))


def test_arbitrary_labels_are_mapped_back():
    X, y = blobs(sep=6.0)
    labels = np.where(y == 1, "premium", "standard")
    svm = PegasosSVM(lambda_=1e-3, n_epochs=10, batch_size=4).fit(X, labels)
    assert set(svm.predict(X)) == {"premium", "standard"}
    assert np.mean(svm.predict(X) == labels) > 0.95
    with pytest.raises(ValueError):
        PegasosSVM().fit(X, np.arange(len(X)) % 3)


def test_training_is_deterministic():
    X, y = blobs()
    a = PegasosSVM(n_epochs=5, batch_size=4, random_state=7).fit(X, y)
    b = PegasosSVM(n_epochs=5, batch_size=4, random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.w, b.w)


@pytest.mark.parametrize("schedule", ["invscaling", "constant"])
def test_other_learning_rate_schedules_train(schedule):
    X, y = blobs(sep=4.0)
    svm = PegasosSVM(lambda_=1e-2, n_epochs=30, batch_size=8, lr_schedule=schedule,
                     eta0=0.5).fit(X, y)
    assert np.mean(svm.predict(X) == y) > 0.95


def test_rff_approximates_the_rbf_kernel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    K = rbf_kernel(X, X, gamma=0.2)
    errors = []
    for D in (16, 256, 4096):
        Z = RandomFourierFeatures(D, gamma=0.2, random_state=1).fit_transform(X)
        errors.append(np.abs(Z @ Z.T - K).mean())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.02


def test_rff_svm_solves_a_nonlinear_problem_the_linear_svm_cannot():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(600, 2))
    y = (np.sum(X ** 2, axis=1) < 0.45).astype(int)          # disc inside a square
    linear = PegasosSVM(lambda_=1e-4, n_epochs=30, batch_size=16).fit(X, y)
    rff = RFFPegasosSVM(gamma=2.0, n_components=300, lambda_=1e-4, n_epochs=30,
                        batch_size=16).fit(X, y)
    assert np.mean(rff.predict(X) == y) > 0.9
    assert np.mean(rff.predict(X) == y) > np.mean(linear.predict(X) == y) + 0.15


def test_iterate_averaging_reduces_the_objective():
    X, y = blobs(n=400, sep=2.0, seed=4)
    last = PegasosSVM(lambda_=1e-3, n_epochs=20, batch_size=8, average=False).fit(X, y)
    avg = PegasosSVM(lambda_=1e-3, n_epochs=20, batch_size=8, average=True).fit(X, y)
    assert avg.history_["objective"][-1] < last.history_["objective"][-1]
    # the returned weights are the averaged ones
    np.testing.assert_allclose(avg.coef_, avg.w[:-1])
