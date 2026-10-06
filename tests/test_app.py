"""Web interface: the serving model and the HTTP API, on SYNTHETIC data."""

import numpy as np
import pytest

from src import data_prep as dp
from tests.fake_binaaz import make_fake_binaaz

fastapi = pytest.importorskip("fastapi", reason="app extras not installed (pip install -r requirements-app.txt)")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.predictor import Listing, PricePredictor  # noqa: E402

FLAT = {"category": "yeni tikili", "area_m2": 90, "rooms": 3, "floor": 7, "total_floors": 16,
        "repair": True, "bill_of_sale": True, "description": "Metroya yaxın, avro təmirli"}


@pytest.fixture(scope="module")
def model():
    cleaned = dp.clean(make_fake_binaaz(2500, seed=5))
    return PricePredictor().fit(cleaned, fast=True, log=lambda *_: None)


@pytest.fixture(scope="module")
def client(model):
    with TestClient(create_app(model=model)) as c:
        yield c


def a_location(model):
    return next(k for k, v in model.gazetteer_.items() if v["lat"] is not None)


def test_prediction_is_consistent(model):
    p = model.predict(Listing(**FLAT, location=a_location(model)))
    assert p.price_low_azn < p.price_azn < p.price_high_azn
    assert p.trees_low_azn <= p.price_azn <= p.trees_high_azn * 1.01
    assert p.price_per_m2_azn == pytest.approx(p.price_azn / 90, rel=0.01)
    assert p.premium == (p.svm_score >= 0)
    assert p.location_used["coordinates_from"] == "typical for location"


def test_factors_add_up_to_the_prediction(model):
    """Path decomposition is exact: baseline + all contributions = forest output."""
    item = Listing(**FLAT, location=a_location(model))
    df, _ = model._frame(item)
    X = model.pre_.transform(dp.make_features(df)[0])
    total = model.baseline_log_ + sum(f["log_effect"] for f in model.explain(X[0], top=999))
    assert total == pytest.approx(np.log(model.predict(item).price_azn), abs=2e-3)   # rounding to 100 AZN


def test_location_brings_its_tags(model):
    """A chosen location fills in its typical metro / landmark tags."""
    loc = next(k for k, v in model.gazetteer_.items() if v.get("tags"))
    df, _ = model._frame(Listing(**FLAT, location=loc))
    assert df["tags"].iloc[0] == model.gazetteer_[loc]["tags"]
    assert any(n.startswith("tags=") for n in model.pre_.feature_names_)


def test_bigger_flat_costs_more(model):
    loc = a_location(model)
    small = model.predict(Listing(**{**FLAT, "area_m2": 45, "rooms": 1}, location=loc))
    big = model.predict(Listing(**{**FLAT, "area_m2": 200, "rooms": 5}, location=loc))
    assert big.price_azn > small.price_azn


def test_save_load_roundtrip(model, tmp_path):
    path = model.save(tmp_path / "m.pkl")
    again = PricePredictor.load(path)
    item = Listing(**FLAT)
    assert again.predict(item) == model.predict(item)


def test_load_rejects_other_pickles(tmp_path):
    import pickle
    path = tmp_path / "x.pkl"
    path.write_bytes(pickle.dumps({"not": "a model"}))
    with pytest.raises(ValueError):
        PricePredictor.load(path)


def test_interval_calibrated_without_test(model):
    """Calibrated on out-of-bag residuals, the band still covers ~80% of test."""
    assert 0.65 <= model.metrics_["interval_coverage"] <= 0.95
    assert "out-of-bag" in model.metrics_["interval_calibration"]


def test_oob_replay_matches_forest():
    """oob_predict replays RandomForest's bootstrap draws; a drift would raise."""
    from src.ensemble import RandomForest
    from app.predictor import conformal_quantiles, oob_predict
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = X[:, 0] + 0.1 * rng.normal(size=300)
    rf = RandomForest("regression", n_estimators=10, max_features=1 / 3, random_state=7).fit(X, y)
    oob = oob_predict(rf, X, y, seed=7)
    assert np.isfinite(oob).mean() > 0.95                     # ~(1-1/e)^10 rows never OOB
    with pytest.raises(RuntimeError):
        oob_predict(rf, X, y, seed=8)                          # wrong seed is caught
    lo, hi = conformal_quantiles(y - np.nan_to_num(oob), 0.8)
    assert lo < 0 < hi


# ------------------------------------------------------------------ HTTP API
def test_health_and_options(client, model):
    assert client.get("/api/health").json() == {"status": "ok", "model_loaded": True}
    opts = client.get("/api/options").json()
    assert {c["value"] for c in opts["categories"]} <= {"yeni tikili", "kohne tikili", "heyet evi/bag evi"}
    assert len(opts["locations"]) == len(model.gazetteer_)
    card = client.get("/api/model").json()
    assert 0 <= card["test_roc_auc"] <= 1 and card["fast"] is True


def test_predict_endpoint(client, model):
    r = client.post("/api/predict", json={**FLAT, "location": a_location(model)})
    assert r.status_code == 200
    body = r.json()
    assert body["price_azn"] > 0 and isinstance(body["premium"], bool)
    assert body["factors"] and all(np.isfinite(f["pct_effect"]) for f in body["factors"])


def test_house_ignores_floors_and_uses_land(client):
    house = {"category": "heyet evi/bag evi", "area_m2": 180, "rooms": 5, "land_area_sot": 6}
    assert client.post("/api/predict", json=house).status_code == 200


@pytest.mark.parametrize("bad", [
    {"area_m2": 2},                                  # below the cleaning rule
    {"floor": 20, "total_floors": 5},                # floor above building
    {"category": "ofis"},                            # out-of-scope category
    {"lat": 40.4},                                   # lat without lng
    {"location": "atlantis"},                        # unknown location
    {"surprise": 1},                                 # unknown field
])
def test_predict_rejects_bad_input(client, bad):
    assert client.post("/api/predict", json={**FLAT, **bad}).status_code == 422


def test_no_model_gives_503(tmp_path):
    with TestClient(create_app(model_path=tmp_path / "missing.pkl")) as c:
        assert c.get("/api/health").json()["model_loaded"] is False
        assert c.post("/api/predict", json=FLAT).status_code == 503


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Price estimator" in r.text
    assert client.get("/static/app.js").status_code == 200
