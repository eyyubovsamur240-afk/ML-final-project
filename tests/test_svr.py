"""Tests for the from-scratch epsilon-insensitive SVR (bonus, src/svr.py)."""

import numpy as np
import pytest
from sklearn.svm import LinearSVR

from src.linear import RidgeRegression
from src.svr import PegasosSVR, RFFPegasosSVR


def linear_data(n=400, d=3, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ np.arange(1, d + 1, dtype=float) + noise * rng.normal(size=n)
    return X, y - y.mean()          # centred: then our problem and LIBLINEAR's coincide


def eps_objective(w, b, X, y, lam, eps):
    """sklearn LinearSVR primal divided by n*C, with C = 1/(n*lambda) (bias regularised)."""
    return 0.5 * lam * (w @ w + b * b) + np.mean(np.maximum(0, np.abs(y - X @ w - b) - eps))


@pytest.mark.parametrize("batch_size", [1, 16, None])
def test_objective_decreases_and_settles(batch_size):
    X, y = linear_data()
    m = PegasosSVR(lambda_=1e-2, epsilon=0.1, n_epochs=40, batch_size=batch_size).fit(X, y)
    J = np.array(m.history_["objective"])
    assert J[-1] < 0.5 * J[0]
    assert J[-1] <= np.min(J[1:]) + 0.02


@pytest.mark.parametrize("eps", [0.0, 0.2])
def test_reaches_the_same_objective_as_liblinear(eps):
    X, y = linear_data(seed=2)
    lam = 1e-2
    ours = PegasosSVR(lambda_=lam, epsilon=eps, n_epochs=300, batch_size=16).fit(X, y)
    sk = LinearSVR(C=1 / (len(y) * lam), epsilon=eps, loss="epsilon_insensitive", dual=True,
                   max_iter=200_000, tol=1e-8, random_state=0).fit(X, y)
    j_ours = eps_objective(ours.coef_, ours.intercept_, X, y, lam, eps)
    j_sk = eps_objective(sk.coef_, sk.intercept_[0], X, y, lam, eps)
    assert j_ours <= j_sk * 1.01
    assert np.corrcoef(ours.predict(X), sk.predict(X))[0, 1] > 0.999


def test_recovers_the_true_line_and_the_intercept():
    X, y = linear_data(noise=0.05, seed=4)
    y = y + 12.0                    # log(price)-sized offset: handled by centring y
    m = PegasosSVR(lambda_=1e-3, epsilon=0.0, n_epochs=200, batch_size=8).fit(X, y)
    np.testing.assert_allclose(m.coef_, [1, 2, 3], atol=0.05)
    assert abs(np.mean(m.predict(X) - y)) < 0.05


def test_wide_tube_ignores_the_data():
    X, y = linear_data()
    m = PegasosSVR(lambda_=1e-2, epsilon=100.0, n_epochs=5).fit(X, y)
    assert np.linalg.norm(m.coef_) < 1e-12           # every residual is inside the tube
    assert m.tube_fraction(X, y) == 1.0


def test_epsilon_loss_is_robust_to_outliers_unlike_least_squares():
    X, y = linear_data(n=500, d=1, noise=0.1, seed=5)
    y_bad = y.copy()
    y_bad[np.argsort(X[:, 0])[-25:]] += 30.0          # 5% gross errors at one end
    svr = PegasosSVR(lambda_=1e-3, epsilon=0.0, n_epochs=200, batch_size=8).fit(X, y_bad)
    ols = RidgeRegression(alpha=0.0).fit(X, y_bad)
    assert abs(svr.coef_[0] - 1.0) < 0.15
    assert abs(ols.coef_[0] - 1.0) > 1.0


def test_rff_svr_fits_a_nonlinear_curve():
    rng = np.random.default_rng(6)
    X = rng.uniform(-3, 3, size=(600, 1))
    y = np.sin(2 * X[:, 0]) + 0.05 * rng.normal(size=600)
    lin = PegasosSVR(lambda_=1e-4, epsilon=0.05, n_epochs=30).fit(X, y)
    rff = RFFPegasosSVR(gamma=1.0, n_components=300, lambda_=3e-4, epsilon=0.05, n_epochs=200,
                        batch_size=8).fit(X, y)
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))   # noqa: E731
    assert rmse(rff.predict(X)) < 0.2 < rmse(lin.predict(X))


def test_is_deterministic_given_the_seed():
    X, y = linear_data()
    a = PegasosSVR(n_epochs=3, random_state=1).fit(X, y).coef_
    b = PegasosSVR(n_epochs=3, random_state=1).fit(X, y).coef_
    np.testing.assert_array_equal(a, b)


def test_offset_schedule_survives_a_tiny_lambda_where_plain_pegasos_does_not():
    """With 1/lambda >> number of steps, plain 1/(lambda t) steps have not settled yet."""
    X, y = linear_data(seed=7)
    kw = dict(lambda_=1e-6, epsilon=0.0, n_epochs=10, batch_size=16)
    offset = PegasosSVR(**kw).fit(X, y)
    plain = PegasosSVR(lr_schedule="pegasos", **kw).fit(X, y)
    rmse = lambda m: float(np.sqrt(np.mean((m.predict(X) - y) ** 2)))   # noqa: E731
    assert rmse(offset) < 0.5 < rmse(plain)
