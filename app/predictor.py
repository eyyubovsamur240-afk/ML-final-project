"""
predictor.py — the serving model behind the web interface.

``PricePredictor`` bundles everything needed to price ONE listing typed in
by a person, using the project's own from-scratch models:

  * Task A (price):  ``RandomForest`` of our CART trees on log(price per m²),
    plus the listing's log(area) — the best Task A model in the report
    (bonus section).
  * Task B (tier):   ``RFFPegasosSVM`` — Pegasos on random Fourier features,
    the best Task B model by ROC-AUC.

Preprocessing is exactly the training pipeline (``make_features`` ->
``Preprocessor`` -> ``Standardizer``), fitted on train+val only. The test
split is scored once at training time to give the interface honest numbers:
held-out metrics, the coverage of the 80% price range (calibrated on the
forest's out-of-bag residuals, not on test), and how often the SVM is right
inside vs outside its margin.

Nothing here learns from the user's input; ``predict`` is a pure function of
the fitted bundle.
"""

from __future__ import annotations

import hashlib
import pickle
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src import data_prep as dp
from src import evaluate as ev
from src.ensemble import RandomForest
from src.svm import RFFPegasosSVM

BUNDLE_VERSION = 1
DEFAULT_MODEL_PATH = config.ROOT / "models" / "predictor.pkl"

# Hyperparameters: the configurations the full run selected on VALIDATION
# data (results/metrics.json -> svm.best_rff, bonus.ensembles). Kept here,
# not re-searched, so training the app takes minutes, not half an hour.
FOREST = dict(n_estimators=config.FOREST_TREES, max_features=1 / 3,
              min_samples_leaf=config.FOREST_MIN_LEAF)
RFF_SVM = dict(gamma=0.001, lambda_=1e-6, n_components=config.RFF_COMPONENTS,
               n_epochs=config.SVM_EPOCHS, batch_size=config.SVM_BATCH,
               average=config.SVM_AVERAGE)
FAST_FOREST = dict(n_estimators=8, max_features=1 / 3, min_samples_leaf=2)
FAST_RFF_SVM = dict(gamma=0.01, lambda_=1e-5, n_components=256, n_epochs=8,
                    batch_size=config.SVM_BATCH, average=True)

# Labels shown in the interface for the three residential categories
# (keys are the cleaned, transliterated values the models were trained on).
CATEGORY_LABELS = {
    "yeni tikili": "New building",
    "kohne tikili": "Old building",
    "heyet evi/bag evi": "House / villa",
}
_LOCATION_KIND = {"m.": "metro", "q.": "settlement", "r.": "district"}

# Model columns -> the plain-language factor shown in "what drove this price".
# One-hot levels ("district=nesimi r.") and missing flags ("rooms_missing")
# are folded into their base column first.
FACTOR_LABELS = {
    "area_m2": "Area", "log_area": "Area", "rooms": "Rooms", "area_per_room": "Area per room",
    "floor": "Floor", "total_floors": "Building height", "floor_ratio": "Floor",
    "is_top_floor": "Floor", "is_first_floor": "Floor", "land_area_sot": "Land",
    "has_land": "Land", "lat": "Exact position", "lng": "Exact position",
    "dist_center_km": "Distance to the centre", "desc_log_len": "Listing text length",
    "repair": "Renovation", "mortgage": "Mortgage", "bill_of_sale": "Kupça",
    "kw_metro": "Listing text", "kw_sea": "Listing text", "kw_furnished": "Listing text",
    "kw_urgent": "Listing text", "kw_euro_reno": "Listing text", "kw_parking": "Listing text",
    "category": "Property type", "building_type": "Building type", "city": "City",
    "district": "District", "location": "Neighbourhood", "tags": "Metro / landmarks nearby",
}
# The Preprocessor puts the numeric features first, so log(area) has a fixed column.
LOG_AREA_COL = dp.NUMERIC_FEATURES.index("log_area")


def location_label(key: str) -> str:
    """'28 may m.' -> '28 May (metro)';  'xirdalan' -> 'Xirdalan'."""
    name, _, suffix = key.rpartition(" ")
    if suffix in _LOCATION_KIND and name:
        return f"{name.title()} ({_LOCATION_KIND[suffix]})"
    return key.title()


@dataclass
class Listing:
    """One listing as a person describes it. ``None`` means 'not known'."""
    category: str
    area_m2: float
    rooms: float | None = None
    floor: float | None = None
    total_floors: float | None = None
    land_area_sot: float | None = None
    location: str | None = None
    lat: float | None = None
    lng: float | None = None
    repair: bool | None = None
    mortgage: bool = False
    bill_of_sale: bool = False
    description: str = ""


@dataclass
class Prediction:
    price_azn: float
    price_low_azn: float
    price_high_azn: float
    price_per_m2_azn: float
    premium: bool
    svm_score: float
    inside_margin: bool
    tier_threshold_azn: float
    models_agree: bool
    trees_low_azn: float = 0.0
    trees_high_azn: float = 0.0
    factors: list[dict] = field(default_factory=list)
    location_used: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def oob_predict(forest: RandomForest, X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """
    Out-of-bag prediction for every training row: the mean over the trees
    whose bootstrap sample left that row out (NaN if every tree saw it).

    ``RandomForest`` does not store its bootstrap indices, so they are
    replayed from its seeded RNG in the same order ``fit`` draws them. Each
    replay is verified: a tree's root value is the mean target of its
    bootstrap sample, so a mismatch means the replay went out of step.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    sums, counts = np.zeros(n), np.zeros(n)
    for tree in forest.trees_:
        idx = rng.integers(0, n, n)
        rng.integers(2**31 - 1)                      # the tree's own seed, drawn next in fit()
        if not np.isclose(tree.tree_value_[0], y[idx].mean()):
            raise RuntimeError("bootstrap replay does not match RandomForest.fit")
        out = np.ones(n, dtype=bool)
        out[idx] = False
        sums[out] += tree.predict(X[out])
        counts[out] += 1
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def conformal_quantiles(resid: np.ndarray, coverage: float = 0.80) -> tuple[float, float]:
    """Equal-tailed split-conformal band: residual quantiles at (1 -+ coverage)/2
    with the (n+1)/n finite-sample correction, so new listings are covered
    with probability >= ``coverage`` if they look like the calibration rows."""
    n = len(resid)
    a = (1.0 - coverage) / 2.0
    hi_level = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
    lo_level = max(0.0, np.floor((n + 1) * a) / n)
    return float(np.quantile(resid, lo_level)), float(np.quantile(resid, hi_level))


def _strip_for_serving(model) -> None:
    """Drop the recursive Node objects and RNGs: ``predict`` only needs the
    flattened arrays, and this keeps the pickle small and shallow."""
    trees = getattr(model, "trees_", [])
    for t in trees:
        t.root = None
        t.__dict__.pop("_rng", None)


class PricePredictor:
    """Fit once with ``fit``, persist with ``save``, serve with ``load`` + ``predict``."""

    # ------------------------------------------------------------------ fit
    def fit(self, cleaned: pd.DataFrame, fast: bool = False, seed: int = config.SEED,
            log=print) -> "PricePredictor":
        """
        ``cleaned`` is the output of ``data_prep.clean``. Uses the project's
        deterministic stratified split: fit on train+val, score test once.
        """
        feats, price, _ = dp.make_features(cleaned)
        strata = dp.price_strata(price)
        tr, va, te = dp.split_indices(len(price), stratify=strata, seed=seed)
        dev = np.sort(np.concatenate([tr, va]))

        self.pre_ = dp.Preprocessor().fit(feats.iloc[dev])
        X_dev = self.pre_.transform(feats.iloc[dev])
        X_te = self.pre_.transform(feats.iloc[te])
        self.scaler_ = dp.Standardizer().fit(X_dev)
        Z_dev, Z_te = self.scaler_.transform(X_dev), self.scaler_.transform(X_te)
        y_dev, y_te = np.log(price[dev]), np.log(price[te])
        tier_dev, self.threshold_ = dp.make_tier_label(price[dev])
        tier_te, _ = dp.make_tier_label(price[te], self.threshold_)

        # Trees learn log(price per m²) and add the row's own log(area) back
        # (config.PER_M2_TARGET, as in experiments.PerM2Target).
        self.per_m2_ = bool(config.PER_M2_TARGET)
        y_fit = y_dev - self._offset(X_dev)
        forest_kw = FAST_FOREST if fast else FOREST
        svm_kw = FAST_RFF_SVM if fast else RFF_SVM
        t0 = time.perf_counter()
        log(f"Fitting random forest ({forest_kw['n_estimators']} trees) on {len(dev):,} listings")
        self.forest_ = RandomForest("regression", random_state=seed, **forest_kw).fit(X_dev, y_fit)
        t1 = time.perf_counter()
        log(f"  done in {t1 - t0:.0f}s. Fitting RFF Pegasos SVM")
        self.svm_ = RFFPegasosSVM(random_state=seed, record_objective=False,
                                  **svm_kw).fit(Z_dev, tier_dev)
        t2 = time.perf_counter()
        log(f"  done in {t2 - t1:.0f}s. Scoring the held-out test split")

        # --- 80% price range: split-conformal on OUT-OF-BAG residuals of the
        # dev rows, so the test split stays untouched and its coverage below
        # is a real check, not true by construction.
        resid_oob = y_fit - oob_predict(self.forest_, X_dev, y_fit, seed)
        resid_oob = resid_oob[np.isfinite(resid_oob)]
        self.interval_ = conformal_quantiles(resid_oob, coverage=0.80)

        # --- honest numbers for the interface, from the untouched test split
        pred_te = self.forest_.predict(X_te) + self._offset(X_te)
        resid = y_te - pred_te
        score_te = self.svm_.decision_function(Z_te)
        tier_pred = (score_te >= 0).astype(int)
        inside = np.abs(score_te) < 1.0
        self.metrics_ = {
            "regression": ev.regression_metrics(y_te, pred_te),
            "classification": ev.classification_metrics(tier_te, tier_pred, score_te),
            "interval_coverage": float(np.mean((resid >= self.interval_[0]) & (resid <= self.interval_[1]))),
            "interval_calibration": f"out-of-bag residuals of {len(resid_oob):,} dev listings",
            "acc_inside_margin": float(np.mean(tier_pred[inside] == tier_te[inside])) if inside.any() else None,
            "acc_outside_margin": float(np.mean(tier_pred[~inside] == tier_te[~inside])) if (~inside).any() else None,
            "share_inside_margin": float(inside.mean()),
            "n_dev": int(len(dev)), "n_test": int(len(te)),
            "fit_seconds": {"forest": t1 - t0, "svm": t2 - t1},
        }
        self.log_area_ref_ = float(np.median(X_dev[:, LOG_AREA_COL])) if self.per_m2_ else 0.0
        self._fit_reference(cleaned.iloc[dev], feats.iloc[dev])
        _strip_for_serving(self.forest_)
        self.meta_ = {"bundle_version": BUNDLE_VERSION, "fast": fast, "seed": seed,
                      "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "python": platform.python_version(),
                      "forest": forest_kw, "svm": svm_kw, "per_m2_target": self.per_m2_}
        return self

    def _offset(self, X: np.ndarray) -> np.ndarray | float:
        """log(area) for a per-m² forest, else 0: forest output + offset = log(price)."""
        return X[:, LOG_AREA_COL] if getattr(self, "per_m2_", False) else 0.0

    def _fit_reference(self, df_dev: pd.DataFrame, feats_dev: pd.DataFrame) -> None:
        """Gazetteer (typical district/city/coordinates per location) and the
        training ranges used to warn about out-of-range inputs."""
        gaz = {}
        for loc in self.pre_.levels_["location"]:
            rows = df_dev[df_dev["location"] == loc]
            mode = lambda s: s.mode().iloc[0] if s.notna().any() else None  # noqa: E731
            lat, lng = rows["lat"].median(), rows["lng"].median()
            gaz[loc] = {
                "label": location_label(loc),
                "district": mode(rows["district"]), "city": mode(rows["city"]),
                "lat": None if np.isnan(lat) else round(float(lat), 5),
                "lng": None if np.isnan(lng) else round(float(lng), 5),
                "n_listings": int(len(rows)),
                "tags": mode(rows["tags"]) if "tags" in rows else None,
            }
        self.gazetteer_ = dict(sorted(gaz.items(), key=lambda kv: kv[1]["label"]))
        self.ranges_ = {c: (float(feats_dev[c].quantile(0.005)), float(feats_dev[c].quantile(0.995)))
                        for c in ("area_m2", "rooms", "total_floors", "land_area_sot")}
        self.categories_ = [c for c in CATEGORY_LABELS if c in self.pre_.levels_["category"]]

    # ------------------------------------------------------------------ predict
    def _frame(self, item: Listing) -> tuple[pd.DataFrame, dict]:
        """Build a one-row frame shaped like ``data_prep.clean`` output."""
        place = self.gazetteer_.get(item.location) if item.location else None
        lat = item.lat if item.lat is not None else (place or {}).get("lat")
        lng = item.lng if item.lng is not None else (place or {}).get("lng")
        land = item.land_area_sot if item.category == "heyet evi/bag evi" else None
        row = {
            "currency": "AZN", "price_azn": np.nan, "area_m2": float(item.area_m2),
            "rooms": item.rooms, "floor": item.floor, "total_floors": item.total_floors,
            "land_area_sot": land, "lat": lat, "lng": lng,
            "repair": None if item.repair is None else float(item.repair),
            "mortgage": float(item.mortgage), "bill_of_sale": float(item.bill_of_sale),
            "category": item.category, "building_type": np.nan,
            "city": (place or {}).get("city", np.nan), "location": item.location or np.nan,
            "district": (place or {}).get("district", np.nan),
            "tags": (place or {}).get("tags") or "",
            "description": item.description or "",
        }
        df = pd.DataFrame([row]).astype({c: float for c in (
            "area_m2", "rooms", "floor", "total_floors", "land_area_sot", "lat", "lng",
            "repair", "mortgage", "bill_of_sale")})
        text = lambda v: v if isinstance(v, str) else None  # noqa: E731
        used = {"location": text(row["location"]), "district": text(row["district"]),
                "city": text(row["city"]),
                "lat": lat, "lng": lng, "coordinates_from": "you" if item.lat is not None
                else ("typical for location" if lat is not None else "not known")}
        return df, used

    def _warnings(self, item: Listing) -> list[str]:
        out = []
        for name, value, label in (("area_m2", item.area_m2, "Area"), ("rooms", item.rooms, "Rooms"),
                                   ("total_floors", item.total_floors, "Floors in the building")):
            lo, hi = self.ranges_[name]
            if value is not None and not lo <= value <= hi:
                out.append(f"{label} {value:g} is outside what the models saw in training "
                           f"({lo:g}–{hi:g}); treat the estimate with care.")
        if item.location and item.location not in self.gazetteer_:
            out.append("Unknown location; it was treated as 'other'.")
        if item.location is None and item.lat is None:
            out.append("No location given; location is the strongest price driver, "
                       "so the estimate is rough.")
        return out

    def predict(self, item: Listing) -> Prediction:
        if item.category not in CATEGORY_LABELS:
            raise ValueError(f"category must be one of {list(CATEGORY_LABELS)}")
        df, used = self._frame(item)
        feats, _, _ = dp.make_features(df)
        X = self.pre_.transform(feats)
        offset = float(np.atleast_1d(self._offset(X))[0])
        per_tree = np.array([t.predict(X)[0] for t in self.forest_.trees_]) + offset
        log_price = float(per_tree.mean())          # == RandomForest.predict
        t_lo, t_hi = np.exp(np.quantile(per_tree, [0.10, 0.90]))
        score = float(self.svm_.decision_function(self.scaler_.transform(X))[0])
        price = float(np.exp(log_price))
        lo, hi = self.interval_
        return Prediction(
            price_azn=round(price, -2), price_low_azn=round(price * np.exp(lo), -2),
            price_high_azn=round(price * np.exp(hi), -2),
            price_per_m2_azn=round(price / item.area_m2),
            premium=bool(score >= 0), svm_score=round(score, 4),
            inside_margin=bool(abs(score) < 1.0),
            tier_threshold_azn=round(self.threshold_, -2),
            models_agree=bool((price > self.threshold_) == (score >= 0)),
            trees_low_azn=round(float(t_lo), -2), trees_high_azn=round(float(t_hi), -2),
            factors=self.explain(X[0]),
            location_used=used, warnings=self._warnings(item),
        )

    @property
    def baseline_log_(self) -> float:
        """The forest's starting point: mean log-price of the training rows
        (each tree's root value, averaged). For a per-m² forest the root is a
        log price per m², so the median listing's log(area) is added."""
        return float(np.mean([t.tree_value_[0] for t in self.forest_.trees_])) \
            + getattr(self, "log_area_ref_", 0.0)

    def explain(self, x: np.ndarray, top: int = 6) -> list[dict]:
        """
        What drove this price: the forest's prediction split exactly into a
        baseline plus one contribution per feature (Saabas, 2014). In each tree,
        every split on the row's path moves the running prediction from the
        parent's mean to the child's mean; that move is credited to the split
        feature. Averaged over trees, baseline + sum(contributions) equals the
        forest's log-price, so each contribution is a multiplicative effect
        on price: exp(c) - 1. Grouped into plain-language factors.
        """
        contrib = np.zeros(len(x))
        for t in self.forest_.trees_:
            node = 0
            while t.tree_feature_[node] >= 0:
                f = t.tree_feature_[node]
                child = t.tree_left_[node] if x[f] <= t.tree_threshold_[node] else t.tree_right_[node]
                contrib[f] += t.tree_value_[child] - t.tree_value_[node]
                node = child
        contrib /= len(self.forest_.trees_)
        if getattr(self, "per_m2_", False):    # the area offset: this flat vs the median one
            contrib[LOG_AREA_COL] += x[LOG_AREA_COL] - self.log_area_ref_
        grouped: dict[str, float] = {}
        for name, c in zip(self.pre_.feature_names_, contrib):
            base = name.split("=")[0].removesuffix("_missing")
            label = FACTOR_LABELS.get(base, base)
            grouped[label] = grouped.get(label, 0.0) + float(c)
        ranked = sorted(grouped.items(), key=lambda kv: -abs(kv[1]))[:top]
        return [{"factor": k, "log_effect": round(v, 4), "pct_effect": round(100 * np.expm1(v), 1)}
                for k, v in ranked if abs(v) > 1e-4]

    # ------------------------------------------------------------------ options
    def options(self) -> dict:
        """Everything the form needs: dropdown values and sensible bounds."""
        return {
            "categories": [{"value": c, "label": CATEGORY_LABELS[c]} for c in self.categories_],
            "typical_price_azn": round(float(np.exp(self.baseline_log_)), -2),
            "locations": [{"value": k, **v} for k, v in self.gazetteer_.items()],
            "ranges": self.ranges_,
            "map_center": list(config.BAKU_CENTER),
        }

    def card(self) -> dict:
        """A small model card for the interface's 'About the models' panel."""
        r, c = self.metrics_["regression"], self.metrics_["classification"]
        return {
            **self.meta_,
            "price_model": ("Random forest of our from-scratch CART trees on log(price per m²), times the area"
                            if getattr(self, "per_m2_", False) else
                            "Random forest of our from-scratch CART trees on log(price)"),
            "tier_model": "RBF-kernel SVM: random Fourier features + our Pegasos solver",
            "test_rmse_log": r["rmse_log"], "test_r2_log": r["r2_log"], "test_mape": r["mape"],
            "test_f1": c["f1"], "test_roc_auc": c["roc_auc"], "test_accuracy": c["accuracy"],
            "interval_coverage": self.metrics_["interval_coverage"],
            "acc_inside_margin": self.metrics_["acc_inside_margin"],
            "acc_outside_margin": self.metrics_["acc_outside_margin"],
            "tier_threshold_azn": self.threshold_,
            "n_dev": self.metrics_["n_dev"], "n_test": self.metrics_["n_test"],
            "data_sha256": self.meta_.get("data_sha256"),
        }

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)                     # atomic: a server never reads half a file
        return path

    @staticmethod
    def load(path: str | Path = DEFAULT_MODEL_PATH) -> "PricePredictor":
        """Load a bundle written by ``save``. Pickles can run code: only load
        files you trained yourself."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, PricePredictor) or obj.meta_.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError(f"{path} is not a v{BUNDLE_VERSION} PricePredictor bundle; retrain it")
        return obj
