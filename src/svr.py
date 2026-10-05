"""
svr.py — BONUS (Problem 8): support-vector REGRESSION from scratch (NumPy only).

The SVM of ``svm.py`` re-used for Task A. A linear SVR minimises the
regularised epsilon-insensitive loss in the primal,

    J(w) = (lambda/2) * ||w||^2 + (1/n) * sum_i max(0, |y_i - w.x_i| - eps),

which is the regression twin of the hinge loss: residuals inside the
"epsilon tube" cost nothing, larger ones cost linearly (so, like an L1 loss,
a few luxury outliers pull the fit much less than they pull least squares).
A sub-gradient with respect to w for one example is

    g_i = -sign(y_i - w.x_i) x_i   if |y_i - w.x_i| > eps,   else 0,

so the Pegasos step from ``svm.py`` carries over unchanged except for the
active set and the sign:

    A_t+ = {i in A_t : |r_i| > eps},   r_i = y_i - w.x_i
    w <- (1 - eta_t lambda) w + (eta_t/k) * sum_{i in A_t+} sign(r_i) x_i
    w <- min(1, R / ||w||) * w         (projection, see below)

Step size. Pegasos' eta_t = 1/(lambda t) is fine for the classifier, whose
predictions only depend on the SIGN of w.x, but for regression the scale of
w is the prediction: with a small lambda the first steps (eta_1 = 1/lambda)
throw w far away and 1/lambda steps are needed before it settles - more than
20 epochs give on this dataset (we measured validation RMSE 17.5 at
lambda = 1e-6). We therefore default to the offset schedule
eta_t = 1/(lambda (t + t0)) with t0 = 1/(lambda eta0), i.e. the first step
is eta0 = 1/mean||x||^2 (one step moves a prediction by about one target
unit) and the tail is still Pegasos' 1/(lambda t) (Bottou, 2012); the plain
schedule is kept as ``lr_schedule="pegasos"``. t-weighted iterate averaging
exactly as in ``PegasosSVM``. The projection radius is the regression analogue of
the classifier's 1/sqrt(lambda): J(w*) <= J(0) = L0 (the loss of predicting
the mean) and J(w) >= (lambda/2)||w||^2, so ||w*|| <= sqrt(2 L0 / lambda).
Projecting onto that ball never cuts off the optimum; it only stops the huge
early steps (eta_1 = 1/lambda) from throwing w far away.

Target and bias. y is centred (y_mean_ is added back in ``predict``), so the
bias folded in as a constant feature only has to learn a small offset and its
light regularisation costs nothing - the same trick as the classifier.
Because log(price) has a scale of ~1, epsilon is in log units: eps = 0.1
means errors below ~10% of the price are free.

Non-linear extension: ``RFFPegasosSVR`` maps x through the same
``RandomFourierFeatures`` (RBF kernel approximation) and runs the linear
solver, like ``RFFPegasosSVM``.

scikit-learn's LinearSVR / SVR are BASELINES only - not used here.
"""

from __future__ import annotations

import numpy as np

from .svm import RandomFourierFeatures


class PegasosSVR:
    def __init__(
        self,
        lambda_: float = 1e-4,       # regularisation strength (C = 1/(n*lambda))
        epsilon: float = 0.1,        # half-width of the insensitive tube (target units)
        n_epochs: int = 20,
        batch_size: int = 32,        # 1 = classic Pegasos, None/n = full batch
        lr_schedule: str = "offset",  # "offset": 1/(lambda (t + t0)) | "pegasos": 1/(lambda t)
        eta0: float | None = None,   # first step for "offset" (None: 1 / mean ||x||^2)
        project: bool = True,
        intercept_scaling: float = 1.0,
        average: bool = True,        # t-weighted iterate averaging
        record_objective: bool = True,
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.eta0 = eta0
        self.project = project
        self.intercept_scaling = intercept_scaling
        self.average = average
        self.record_objective = record_objective
        self.random_state = random_state

    def _augment(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([X, np.full((len(X), 1), self.intercept_scaling)])

    def objective(self, Xa, yc, w=None) -> tuple[float, float]:
        """(J(w), mean epsilon-insensitive loss) on augmented data and centred y."""
        w = self.w if w is None else w
        loss = float(np.mean(np.maximum(0.0, np.abs(yc - Xa @ w) - self.epsilon)))
        return 0.5 * self.lambda_ * float(w @ w) + loss, loss

    def fit(self, X, y):
        Xa = self._augment(X)
        y = np.asarray(y, dtype=float)
        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_
        n, d = Xa.shape
        k = n if self.batch_size in (None, 0) else min(int(self.batch_size), n)
        rng = np.random.default_rng(self.random_state)
        # J(w*) <= J(0) = L0 and J(w) >= (lambda/2)||w||^2  =>  ||w*|| <= sqrt(2 L0 / lambda)
        loss0 = float(np.mean(np.maximum(0.0, np.abs(yc) - self.epsilon)))
        radius = np.sqrt(2.0 * max(loss0, 1e-12) / self.lambda_)
        if self.lr_schedule == "pegasos":
            t0 = 0.0
        else:
            eta0 = self.eta0 or 1.0 / float(np.mean((Xa * Xa).sum(axis=1)))
            t0 = 1.0 / (self.lambda_ * eta0)          # so that eta_1 ~= eta0
        w = np.zeros(d)
        w_bar = np.zeros(d)
        self.w = w
        self.history_ = {"epoch": [], "objective": [], "loss": []}
        if self.record_objective:
            self._record(0, Xa, yc)

        t = 0
        for epoch in range(1, self.n_epochs + 1):
            perm = rng.permutation(n)
            for start in range(0, n, k):
                batch = perm[start:start + k]
                t += 1
                eta = 1.0 / (self.lambda_ * (t + t0))
                Xb = Xa[batch]
                r = yc[batch] - Xb @ w
                active = np.abs(r) > self.epsilon            # outside the tube
                w *= (1.0 - eta * self.lambda_)
                if active.any():
                    w += (eta / len(batch)) * (np.sign(r[active]) @ Xb[active])
                if self.project:
                    norm = np.linalg.norm(w)
                    if norm > radius:
                        w *= radius / norm
                if self.average:
                    rho = 2.0 / (t + 1.0)
                    w_bar *= 1.0 - rho
                    w_bar += rho * w
            self.w = w_bar if self.average else w
            if self.record_objective:
                self._record(epoch, Xa, yc)

        self.n_steps_ = t
        self.w = (w_bar if self.average else w).copy()
        self.coef_ = self.w[:-1].copy()
        self.intercept_ = float(self.w[-1] * self.intercept_scaling + self.y_mean_)
        return self

    def _record(self, epoch, Xa, yc):
        J, loss = self.objective(Xa, yc)
        self.history_["epoch"].append(epoch)
        self.history_["objective"].append(J)
        self.history_["loss"].append(loss)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def tube_fraction(self, X, y) -> float:
        """Share of rows predicted within +-epsilon (the non-'support-vector' rows)."""
        return float(np.mean(np.abs(np.asarray(y, dtype=float) - self.predict(X)) <= self.epsilon))


class RFFPegasosSVR:
    """Non-linear SVR = RandomFourierFeatures (RBF kernel) followed by linear PegasosSVR."""

    def __init__(self, gamma: float = 0.01, n_components: int = 1024, lambda_: float = 1e-5,
                 epsilon: float = 0.1, n_epochs: int = 20, batch_size: int = 32,
                 random_state: int = 42, **svr_kwargs):
        self.gamma = gamma
        self.n_components = n_components
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.svr_kwargs = svr_kwargs

    def fit(self, X, y):
        self.rff_ = RandomFourierFeatures(self.n_components, self.gamma, self.random_state).fit(X)
        self.svr_ = PegasosSVR(lambda_=self.lambda_, epsilon=self.epsilon, n_epochs=self.n_epochs,
                               batch_size=self.batch_size, random_state=self.random_state,
                               **self.svr_kwargs).fit(self.rff_.transform(X), y)
        return self

    def predict(self, X):
        return self.svr_.predict(self.rff_.transform(X))
