"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

Pegasos (Shalev-Shwartz et al., 2011) minimises the regularised hinge-loss
primal by stochastic sub-gradient descent — no QP / SMO solver needed:

    J(w) = (lambda/2) * ||w||^2 + (1/n) * sum_i max(0, 1 - y_i w.x_i),   y_i in {-1,+1}

Mini-batch update at step t with batch A_t (size k) and step size eta_t:

    A_t+ = {i in A_t : y_i w.x_i < 1}                       (hinge active)
    w <- (1 - eta_t*lambda) w + (eta_t/k) * sum_{i in A_t+} y_i x_i
    w <- min(1, (1/sqrt(lambda)) / ||w||) * w               (optional projection)

batch_size=1 is the classic Pegasos, batch_size=n is full-batch sub-gradient
descent. Learning-rate schedules: "pegasos" eta_t = 1/(lambda t) (default,
from the paper), "invscaling" eta0/sqrt(t), "constant" eta0.

Iterate averaging (``average=True``): the LAST Pegasos iterate is noisy,
especially for small lambda where the 1/(lambda t) steps stay large for a
long time. We optionally return the weighted average of the iterates with
weights proportional to t,
    w_bar_t = (1 - 2/(t+1)) w_bar_{t-1} + 2/(t+1) w_t,
which keeps the optimal O(1/(lambda T)) rate for this strongly convex
objective (Lacoste-Julien, Schmidt & Bach, 2012) and is one extra line.

Bias: as suggested in the brief we fold b into w by appending a constant
feature (value ``intercept_scaling``) to every x. The bias is therefore
lightly regularised; with standardised features and a median-split
(balanced) label the optimal b is close to 0, so this costs nothing and keeps
the update a single line. The margin ||w|| and the support vectors are
reported for the feature weights only (bias excluded).

Non-linear extension: RANDOM FOURIER FEATURES (Rahimi & Recht, 2007).
phi(x) = sqrt(2/D) cos(W x + c) with W ~ N(0, 2*gamma*I), c ~ U[0, 2pi]
gives phi(x).phi(z) ~= exp(-gamma ||x - z||^2), the RBF kernel. We then run
the SAME linear Pegasos on phi(X). Chosen over kernelised Pegasos because
training/prediction cost is O(n D) instead of O(n * #support vectors), which
matters with tens of thousands of listings, and it re-uses the tested linear
solver unchanged.

scikit-learn's SVC / SGDClassifier are BASELINES only — not used here.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,       # regularisation strength (C = 1/(n*lambda))
        n_epochs: int = 20,          # passes over the data
        batch_size: int = 1,         # 1 = classic Pegasos, None/n = full batch
        lr_schedule: str = "pegasos",  # "pegasos" | "invscaling" | "constant"
        eta0: float = 0.1,           # base step for invscaling / constant
        project: bool = True,        # projection onto the ball ||w|| <= 1/sqrt(lambda)
        intercept_scaling: float = 1.0,
        average: bool = False,       # t-weighted iterate averaging (see module doc)
        record_objective: bool = True,
        random_state: int = 42,
    ):
        if lr_schedule not in ("pegasos", "invscaling", "constant"):
            raise ValueError("lr_schedule must be 'pegasos', 'invscaling' or 'constant'")
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.eta0 = eta0
        self.project = project
        self.intercept_scaling = intercept_scaling
        self.average = average
        self.record_objective = record_objective
        self.random_state = random_state
        self.w = None                # augmented weight vector [coef, bias/scaling]
        self.b = 0.0

    # ------------------------------------------------------------- labels
    def _to_pm1(self, y):
        """Map any 2-class labels to {-1, +1}; classes_[1] is the +1 class."""
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("PegasosSVM is a binary classifier")
        return np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)

    def _augment(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([X, np.full((len(X), 1), self.intercept_scaling)])

    def _eta(self, t: int) -> float:
        if self.lr_schedule == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.lr_schedule == "invscaling":
            return self.eta0 / np.sqrt(t)
        return self.eta0

    def objective(self, X, y_pm1) -> tuple[float, float]:
        """(J(w), mean hinge loss) on augmented data, the quantity being minimised."""
        margins = y_pm1 * (X @ self.w)
        hinge = float(np.mean(np.maximum(0.0, 1.0 - margins)))
        return 0.5 * self.lambda_ * float(self.w @ self.w) + hinge, hinge

    # ------------------------------------------------------------- training
    def fit(self, X, y):
        """
        Run Pegasos. Each epoch shuffles the rows (seeded) and walks through
        them in mini-batches; the objective is recorded after every epoch.
        """
        Xa = self._augment(X)
        y_pm = self._to_pm1(y)
        n, d = Xa.shape
        k = n if self.batch_size in (None, 0) else min(int(self.batch_size), n)
        rng = np.random.default_rng(self.random_state)
        w = np.zeros(d)
        w_bar = np.zeros(d)            # t-weighted average of the iterates
        radius = 1.0 / np.sqrt(self.lambda_)
        self.history_ = {"epoch": [], "step": [], "objective": [], "hinge": [], "w_norm": []}
        self.w = w
        if self.record_objective:
            self._record(0, 0, Xa, y_pm)

        t = 0
        for epoch in range(1, self.n_epochs + 1):
            perm = rng.permutation(n)
            for start in range(0, n, k):
                batch = perm[start:start + k]
                t += 1
                eta = self._eta(t)
                Xb, yb = Xa[batch], y_pm[batch]
                active = yb * (Xb @ w) < 1.0                    # hinge-active samples
                w *= (1.0 - eta * self.lambda_)
                if active.any():
                    w += (eta / len(batch)) * (yb[active] @ Xb[active])
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
                self._record(epoch, t, Xa, y_pm)

        self.n_steps_ = t
        self.w = w_bar.copy() if self.average else w
        self.coef_ = self.w[:-1].copy()
        self.intercept_ = float(self.w[-1] * self.intercept_scaling)
        self.b = self.intercept_
        return self

    def _record(self, epoch, step, Xa, y_pm):
        J, hinge = self.objective(Xa, y_pm)
        self.history_["epoch"].append(epoch)
        self.history_["step"].append(step)
        self.history_["objective"].append(J)
        self.history_["hinge"].append(hinge)
        self.history_["w_norm"].append(float(np.linalg.norm(self.w[:-1])))

    # ------------------------------------------------------------- inference
    def decision_function(self, X):
        """Signed score  w.x + b  for each row."""
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict(self, X):
        """Map sign(w.x + b) back to the original labels."""
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    # ------------------------------------------------------------- margin
    @property
    def w_norm(self) -> float:
        return float(np.linalg.norm(self.coef_))

    @property
    def margin_width(self) -> float:
        """Geometric margin width 2/||w|| (infinite if w = 0)."""
        return 2.0 / self.w_norm if self.w_norm > 0 else np.inf

    def functional_margins(self, X, y):
        """y_i (w.x_i + b) with y in {-1,+1}."""
        y_pm = np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)
        return y_pm * self.decision_function(X)

    def support_vectors(self, X, y, tol: float = 1e-9):
        """Indices of the support vectors: points on or inside the margin, y f(x) <= 1."""
        return np.flatnonzero(self.functional_margins(X, y) <= 1.0 + tol)


class RandomFourierFeatures:
    """phi(x) = sqrt(2/D) cos(W x + c),  W ~ N(0, 2 gamma I),  c ~ U[0, 2 pi]."""

    def __init__(self, n_components: int = 1024, gamma: float = 0.1, random_state: int = 42,
                 dtype=np.float64):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.dtype = dtype

    def fit(self, X):
        d = np.asarray(X).shape[1]
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.n_components))
        self.c_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        Z = np.asarray(X, dtype=float) @ self.W_ + self.c_
        return (np.sqrt(2.0 / self.n_components) * np.cos(Z)).astype(self.dtype)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def rbf_kernel(A, B, gamma: float):
    """Exact RBF kernel matrix exp(-gamma ||a - b||^2) (for checking the RFF approximation)."""
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


class RFFPegasosSVM:
    """Non-linear SVM = RandomFourierFeatures followed by linear PegasosSVM."""

    def __init__(self, gamma: float = 0.1, n_components: int = 1024, lambda_: float = 1e-4,
                 n_epochs: int = 20, batch_size: int = 32, random_state: int = 42, **svm_kwargs):
        self.gamma = gamma
        self.n_components = n_components
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.svm_kwargs = svm_kwargs

    def fit(self, X, y):
        self.rff_ = RandomFourierFeatures(self.n_components, self.gamma, self.random_state).fit(X)
        self.svm_ = PegasosSVM(lambda_=self.lambda_, n_epochs=self.n_epochs,
                               batch_size=self.batch_size, random_state=self.random_state,
                               **self.svm_kwargs).fit(self.rff_.transform(X), y)
        self.classes_ = self.svm_.classes_
        return self

    def decision_function(self, X):
        return self.svm_.decision_function(self.rff_.transform(X))

    def predict(self, X):
        return self.svm_.predict(self.rff_.transform(X))
