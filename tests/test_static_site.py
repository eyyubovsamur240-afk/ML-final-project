"""Static site export: the browser port of the predictor must give the Python answers (SYNTHETIC data)."""

import base64
import gzip
import json
import shutil

import numpy as np
import pytest

from app.export_static import (encode_forest, export, flatten_forest, pack_sections, random_listings,
                               rebuild_right, check_js)
from app.predictor import PricePredictor
from src import data_prep as dp
from tests.fake_binaaz import make_fake_binaaz


@pytest.fixture(scope="module")
def model():
    cleaned = dp.clean(make_fake_binaaz(2500, seed=11))
    return PricePredictor().fit(cleaned, fast=True, log=lambda *_: None)


@pytest.fixture(scope="module")
def site(model, tmp_path_factory):
    out = tmp_path_factory.mktemp("export") / "site"
    export(model, out)
    return out


def unpack(raw: bytes, layout: list[dict]) -> dict:
    """Python mirror of predictor.js unpack(): undo the byte shuffle."""
    types = {"f64": "<f8", "i16": "<i2", "i32": "<i4", "u32": "<u4", "u16": "<u2"}
    out = {}
    for s in layout:
        dt = np.dtype(types[s["dtype"]])
        blob = np.frombuffer(raw, np.uint8, s["count"] * dt.itemsize, s["offset"])
        if s["shuffled"]:
            blob = blob.reshape(dt.itemsize, s["count"]).T.copy()
        out[s["name"]] = np.frombuffer(blob.tobytes(), dt)
    return out


def test_rebuild_right_matches_every_tree(model):
    for t in model.forest_.trees_:
        internal = t.tree_feature_ >= 0
        assert np.array_equal(rebuild_right(t.tree_feature_)[internal], t.tree_right_[internal])


def test_pack_and_dictionary_coding_are_exact(model):
    feat, thr, val, _ = flatten_forest(model)
    core, explain = encode_forest(feat, thr, val, len(model.pre_.feature_names_))
    raw, layout = pack_sections(core + explain)
    sec = unpack(raw, layout)
    leaf = feat < 0
    assert np.array_equal(sec["feat"], feat)
    got_thr = sec["thr_dict"][sec["thr_off"][feat[~leaf]] + sec["thr_idx"]]
    assert np.array_equal(got_thr.view(np.uint64), thr.view(np.uint64))
    assert np.array_equal(sec["leaf_dict"][sec["leaf_idx"]].view(np.uint64), val[leaf].view(np.uint64))
    assert np.array_equal(sec["node_val"].view(np.uint64), val[~leaf].view(np.uint64))


def test_export_writes_a_self_contained_site(site, model):
    meta = json.loads((site / "model.json").read_text())
    assert meta["forest"]["n_trees"] == len(model.forest_.trees_)
    assert meta["preprocess"]["feature_names"] == model.pre_.feature_names_
    for stream in meta["binary"].values():
        packed = b"".join(base64.b64decode((site / f).read_bytes()) for f in stream["files"])
        assert len(packed) == stream["bytes"]
        assert len(gzip.decompress(packed)) == stream["raw_bytes"]
    page = (site / "index.html").read_text()
    assert "<title>" in page and 'src="predictor.js"' not in page and "class Estimator" in page
    assert "<html" not in page            # the artifact host adds the document skeleton


def test_random_listings_are_valid_inputs(model):
    for item in random_listings(model, 50, seed=3):
        assert item["category"] in model.categories_
        if item["floor"] is not None and item["total_floors"] is not None:
            assert item["floor"] <= item["total_floors"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_javascript_port_matches_python(site, model):
    res = check_js(model, site, n=150, seed=1)
    assert res["failures"] == 0, res["first_failures"]
    # identical except, rarely, a last-bit libm difference at a rounding tie
    assert res["identical"] >= 0.97 * res["cases"]
