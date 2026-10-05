"""Tests for parsing, cleaning, leakage-free preprocessing and splitting."""

import numpy as np
import pandas as pd
import pytest

from src import data_prep as dp
from tests.fake_binaaz import make_fake_binaaz


# --------------------------------------------------------------------------- parsers
@pytest.mark.parametrize("raw, expected", [
    ("85 m²", 85.0), ("1 250 000", 1_250_000.0), ("85,5", 85.5), ("1,250,000", 1_250_000.0),
    ("120000 AZN", 120_000.0), (" 300 000", 300_000.0), (42, 42.0),
])
def test_parse_number(raw, expected):
    assert dp.parse_number(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "yoxdur", np.nan])
def test_parse_number_missing(raw):
    assert np.isnan(dp.parse_number(raw))


def test_parse_floor_land_yes_no_currency():
    assert dp.parse_floor("5 / 9") == (5.0, 9.0)
    f, t = dp.parse_floor("3")
    assert f == 3.0 and np.isnan(t)
    assert dp.parse_land_sot("6 sot") == 6.0
    assert dp.parse_land_sot("600 m²") == 6.0
    assert dp.parse_land_sot("0.5 ha") == 50.0
    assert dp.parse_yes_no("var") == 1.0 and dp.parse_yes_no("yoxdur") == 0.0
    assert np.isnan(dp.parse_yes_no(None))
    assert dp.parse_currency("USD") == "USD"
    assert dp.parse_currency(None, "120 000 $") == "USD"
    assert dp.parse_currency(None, "120 000") == "AZN"


def test_normalize_name_handles_azerbaijani():
    assert dp.normalize_name("Otaq sayı") == "otaq_sayi"
    assert dp.normalize_name("Binanın növü") == "binanin_novu"
    assert dp.normalize_name("İpoteka") == "ipoteka"
    assert dp.normalize_name("Çıxarış") == "cixaris"
    assert dp.normalize_name("Mərtəbə") == "mertebe"


def test_attributes_blob_fallback():
    raw = pd.DataFrame({
        "price": ["100 000", "200 000"],
        "attributes": ['{"Sahə": "50 m²", "Otaq sayı": "2", "Mərtəbə": "3 / 9"}',
                       "Sahə: 80 m²; Otaq sayı: 3; Mərtəbə: 7 / 16"],
    })
    df = dp.clean(raw)
    assert df["area_m2"].tolist() == [50.0, 80.0]
    assert df["rooms"].tolist() == [2.0, 3.0]
    assert df["total_floors"].tolist() == [9.0, 16.0]


# --------------------------------------------------------------------------- cleaning
@pytest.fixture(scope="module")
def cleaned():
    log = dp.CleaningLog()
    return dp.clean(make_fake_binaaz(3000, seed=1), log), log


def test_clean_drops_leakage_identifier_and_meta_columns(cleaned):
    df, log = cleaned
    assert {"unit_price", "total_price"} <= set(log.dropped_columns["leakage"])
    assert {"owner_name", "address"} <= set(log.dropped_columns["identifiers"])
    assert "vip" in log.dropped_columns["listing_meta"]
    assert not any("price" in c and c != "price_azn" for c in df.columns)


def test_clean_rules_hold(cleaned):
    df, log = cleaned
    counts = [n for _, n in log.steps]
    assert counts == sorted(counts, reverse=True)           # every step only removes rows
    assert df["price_azn"].between(*dp.config.PRICE_RANGE_AZN).all()
    assert df["area_m2"].between(*dp.config.AREA_RANGE_M2).all()
    assert not df.duplicated().any()
    ok = df["floor"].isna() | df["total_floors"].isna() | (df["floor"] <= df["total_floors"])
    assert ok.all()
    assert set(df["currency"]) <= {"AZN", "USD"}


def test_usd_prices_are_converted(cleaned):
    df, _ = cleaned
    usd = df[df["currency"] == "USD"]
    azn = df[df["currency"] == "AZN"]
    # same generative model -> similar AZN price per m² after conversion
    ratio = (usd["price_azn"] / usd["area_m2"]).median() / (azn["price_azn"] / azn["area_m2"]).median()
    assert 0.8 < ratio < 1.25


# --------------------------------------------------------------------------- features
def test_make_features_and_preprocessor_no_leakage(cleaned):
    df, _ = cleaned
    feats, y, cols = dp.make_features(df)
    assert len(feats) == len(y) and cols == list(feats.columns)
    tr, va, te = dp.split_indices(len(y), stratify=dp.price_strata(y))
    pre = dp.Preprocessor().fit(feats.iloc[tr])
    X_tr, X_va = pre.transform(feats.iloc[tr]), pre.transform(feats.iloc[va])
    assert X_tr.shape[1] == X_va.shape[1] == len(pre.feature_names_)
    assert not np.isnan(X_tr).any() and not np.isnan(X_va).any()
    # imputation medians come from train rows only
    assert pre.medians_["rooms"] == pytest.approx(feats.iloc[tr]["rooms"].median())
    # a level never seen in train goes to "<col>=other"
    unseen = feats.iloc[va[:1]].copy()
    unseen["location"] = "never-seen district"
    row = pre.transform(unseen)[0]
    assert row[pre.feature_names_.index("location=other")] == 1.0


# --------------------------------------------------------------------------- split / label
def test_split_is_disjoint_complete_deterministic_and_stratified():
    rng = np.random.default_rng(0)
    price = np.exp(rng.normal(12, 0.6, 5000))
    strata = dp.price_strata(price)
    a = dp.split_indices(5000, stratify=strata, seed=3)
    b = dp.split_indices(5000, stratify=strata, seed=3)
    for x, z in zip(a, b):
        np.testing.assert_array_equal(x, z)
    tr, va, te = a
    assert len(set(tr) | set(va) | set(te)) == 5000
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert abs(len(te) / 5000 - 0.15) < 0.01
    y_tr, thr = dp.make_tier_label(price[tr])
    for idx in (va, te):
        tier, _ = dp.make_tier_label(price[idx], thr)
        assert abs(tier.mean() - 0.5) < 0.03


def test_tier_threshold_comes_from_train_only():
    y_tr = np.array([1, 2, 3, 4, 100.0])
    tier_tr, thr = dp.make_tier_label(y_tr)
    assert thr == 3.0
    tier_te, thr2 = dp.make_tier_label(np.array([2.5, 3.5]), thr)
    assert thr2 == thr and tier_te.tolist() == [0, 1]
    assert tier_tr.tolist() == [0, 0, 0, 1, 1]


def test_standardize_uses_train_statistics():
    rng = np.random.default_rng(0)
    X_tr, X_te = rng.normal(5, 3, (100, 3)), rng.normal(9, 3, (50, 3))
    Z_tr, Z_te = dp.standardize(X_tr, X_te)
    np.testing.assert_allclose(Z_tr.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Z_tr.std(0), 1)
    np.testing.assert_allclose(Z_te, (X_te - X_tr.mean(0)) / X_tr.std(0))
    const = dp.standardize(np.ones((5, 2)))
    assert np.isfinite(const).all()


def test_stratified_kfold_partitions_rows():
    strata = np.repeat(np.arange(5), 40)
    seen = []
    for tr, te in dp.stratified_kfold(strata, k=4):
        assert not set(tr) & set(te)
        seen.extend(te)
    assert sorted(seen) == list(range(200))
