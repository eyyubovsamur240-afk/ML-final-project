"""
Export the trained web-app bundle as a static site that runs the models in the
browser, so the estimator can be opened from any device without a Python server.

    python -m app.train                        # writes models/predictor.pkl
    python -m app.export_static                # writes site/ (index.html + model files)
    python -m app.export_static --check 500    # also compare JS vs Python on 500 listings

site/ holds:
  index.html   the page (app/static_site/index.html with ui.js and predictor.js inlined)
  model.json   preprocessing, scaler, gazetteer, model card, binary layout
  model-*.txt  gzip stream (base64): the forest (pre-order nodes, exact
               dictionary-coded thresholds and leaf values) and the RFF SVM's weights
  explain-*.txt  gzip stream (base64): internal node means, only needed for
               "what drove this price"; the page fetches it after the first estimate

Nothing in site/ is a listing: the map layer is a grid of listing COUNTS per cell
(training rows only, cells with fewer than MIN_CELL listings dropped).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src import config
from src import data_prep as dp

from .predictor import (CATEGORY_LABELS, DEFAULT_MODEL_PATH, FACTOR_LABELS, LOG_AREA_COL, Listing,
                        PricePredictor, _sha256)

HERE = Path(__file__).resolve().parent / "static_site"
DEFAULT_OUT = config.ROOT / "site"
FORMAT_VERSION = 1
CHUNK_BYTES = 9 * 2**20           # 12 MB once base64-encoded: under the 15 MB per-file limit of static hosts
MAP_BOX = {"lat": (40.25, 40.65), "lng": (49.45, 50.40)}   # Absheron peninsula: Baku, Sumqayit, Xirdalan
MAP_CELL_DEG = 0.004              # ~0.35 x 0.45 km
MIN_CELL = 3                      # suppress cells with fewer listings than this


# ----------------------------------------------------------------- binary model
def flatten_forest(model: PricePredictor):
    """
    All trees in one pre-order node list. The JS walk relies on the layout
    ``left child = node + 1``; right children are rebuilt from the leaf flags,
    so they are not stored. Both facts are checked here, per tree.
    """
    feats, thrs, vals, offsets = [], [], [], []
    start = 0
    for t in model.forest_.trees_:
        f, left, right = t.tree_feature_, t.tree_left_, t.tree_right_
        internal = np.flatnonzero(f >= 0)
        if not np.array_equal(left[internal], internal + 1):
            raise ValueError("tree is not stored in pre-order (left child != node + 1)")
        if not np.array_equal(rebuild_right(f)[internal], right[internal]):
            raise ValueError("right children cannot be rebuilt from the pre-order leaf flags")
        if f.max() >= np.iinfo(np.int16).max:
            raise ValueError("too many features for int16 node features")
        offsets.append(start)
        start += len(f)
        feats.append(f.astype("<i2"))
        thrs.append(t.tree_threshold_[internal].astype("<f8"))
        vals.append(t.tree_value_.astype("<f8"))
    return np.concatenate(feats), np.concatenate(thrs), np.concatenate(vals), offsets


def rebuild_right(feature: np.ndarray) -> np.ndarray:
    """Right child of each internal node of ONE pre-order tree (-1 for leaves):
    after a leaf, the next node is the right child of the deepest open split.
    predictor.js does the same walk."""
    right = np.full(len(feature), -1)
    open_splits = []
    for i, f in enumerate(feature):
        if i > 0 and feature[i - 1] < 0:
            right[open_splits.pop()] = i
        if f >= 0:
            open_splits.append(i)
    return right


def encode_forest(feat: np.ndarray, thr: np.ndarray, val: np.ndarray, n_features: int):
    """
    Exact dictionary coding of the node arrays (decoded bit-for-bit in predictor.js):
      * thresholds: one sorted dictionary per feature (most features are 0/1 and
        only ever split at 0.5) + each split's index into its feature's dictionary;
      * leaf values: few distinct values (a one-listing leaf holds that listing's
        own target), so a dictionary + uint16/uint32 index;
      * internal node values (only used by the explanation) are kept as float64
        in a second file the page downloads after the first estimate is shown.
    """
    leaf = feat < 0
    split_feat = feat[~leaf]
    dicts, thr_idx = [], np.zeros(len(thr), np.uint32)
    thr_off = np.zeros(n_features + 1, np.int32)
    for f in range(n_features):
        mask = split_feat == f
        uniq = np.unique(thr[mask])
        thr_idx[mask] = np.searchsorted(uniq, thr[mask])
        dicts.append(uniq)
        thr_off[f + 1] = thr_off[f] + len(uniq)
    thr_dict = np.concatenate(dicts)
    leaf_dict, leaf_idx = np.unique(val[leaf], return_inverse=True)
    leaf_idx = leaf_idx.astype(np.uint16 if len(leaf_dict) <= 2**16 else np.uint32)
    # the decoding must give back the exact float64 bits
    same = lambda a, b: np.array_equal(a.view(np.uint64), b.view(np.uint64))  # noqa: E731
    if not (same(thr_dict[thr_off[split_feat] + thr_idx], thr) and same(leaf_dict[leaf_idx], val[leaf])):
        raise AssertionError("dictionary coding is not exact")
    core = [("feat", feat), ("thr_off", thr_off), ("thr_dict", thr_dict), ("thr_idx", thr_idx),
            ("leaf_dict", leaf_dict), ("leaf_idx", leaf_idx)]
    return core, [("node_val", val[~leaf])]


def pack_sections(sections: list[tuple[str, np.ndarray]]) -> tuple[bytes, list[dict]]:
    """
    Concatenate arrays, byte-shuffled (byte k of every value stored together),
    which lets gzip squeeze the near-constant exponent bytes of float64 data.
    """
    names = {"<f8": "f64", "<i2": "i16", "<i4": "i32", "<u4": "u32", "<u2": "u16"}
    parts, layout, offset = [], [], 0
    for name, arr in sections:
        dtype = arr.dtype.newbyteorder("<").str
        arr = np.ascontiguousarray(arr, dtype=dtype)
        raw = np.frombuffer(arr.tobytes(), np.uint8)
        width = arr.dtype.itemsize
        if width > 1:
            raw = raw.reshape(-1, width).T
        blob = raw.tobytes()
        layout.append({"name": name, "dtype": names[dtype], "count": int(arr.size),
                       "offset": offset, "shuffled": width > 1})
        parts.append(blob)
        offset += len(blob)
    return b"".join(parts), layout


# ----------------------------------------------------------------- map layer
def density_grid(model: PricePredictor, data: Path | None):
    """Listing counts per map cell over the TRAINING rows of the bundle's own split."""
    if data is None:
        return None
    cleaned = dp.clean(dp.load_raw(data))
    if model.meta_.get("fast") and len(cleaned) > config.FAST["max_rows"]:      # as app.train does
        cleaned = cleaned.sample(n=config.FAST["max_rows"], random_state=config.SEED).reset_index(drop=True)
    _, price, _ = dp.make_features(cleaned)
    tr, va, _ = dp.split_indices(len(price), stratify=dp.price_strata(price), seed=model.meta_["seed"])
    dev = cleaned.iloc[np.concatenate([tr, va])]
    lat, lng = dev["lat"].to_numpy(float), dev["lng"].to_numpy(float)
    (la0, la1), (lo0, lo1) = MAP_BOX["lat"], MAP_BOX["lng"]
    rows, cols = round((la1 - la0) / MAP_CELL_DEG), round((lo1 - lo0) / MAP_CELL_DEG)
    ok = np.isfinite(lat) & np.isfinite(lng) & (lat >= la0) & (lat < la1) & (lng >= lo0) & (lng < lo1)
    r = ((lat[ok] - la0) / MAP_CELL_DEG).astype(int).clip(0, rows - 1)
    c = ((lng[ok] - lo0) / MAP_CELL_DEG).astype(int).clip(0, cols - 1)
    counts = np.zeros((rows, cols), int)
    np.add.at(counts, (r, c), 1)
    cells = [[int(i), int(j), int(counts[i, j])] for i, j in zip(*np.nonzero(counts >= MIN_CELL))]
    return {"box": MAP_BOX, "cell_deg": MAP_CELL_DEG, "rows": rows, "cols": cols,
            "min_cell": MIN_CELL, "cells": cells, "n_listings": int(ok.sum())}


# ----------------------------------------------------------------- export
def build_meta(model: PricePredictor, streams: dict, offsets: list[int], total_nodes: int,
               density) -> dict:
    pre = model.pre_
    rff, lin = model.svm_.rff_, model.svm_.svm_
    card = {k: v for k, v in model.card().items() if k not in ("python",)}
    return {
        "format_version": FORMAT_VERSION,
        "preprocess": {
            "numeric": pre.numeric, "binary": pre.binary, "categorical": pre.categorical,
            "multilabel": pre.multilabel, "indicator_cols": pre.indicator_cols_,
            "medians": {k: float(v) for k, v in pre.medians_.items()},
            "levels": pre.levels_, "labels": pre.labels_, "feature_names": pre.feature_names_,
        },
        "scaler": {"mean": model.scaler_.mean_.tolist(), "scale": model.scaler_.scale_.tolist()},
        "constants": {
            "baku_center": list(config.BAKU_CENTER), "description_keywords": dp.DESCRIPTION_KEYWORDS,
            "per_m2": bool(getattr(model, "per_m2_", False)), "log_area_col": LOG_AREA_COL,
            "log_area_ref": float(getattr(model, "log_area_ref_", 0.0)), "factor_labels": FACTOR_LABELS,
        },
        "limits": {"area": list(config.AREA_RANGE_M2), "rooms": list(config.ROOMS_RANGE),
                   "max_floors": config.MAX_TOTAL_FLOORS, "lat": list(config.LAT_RANGE),
                   "lng": list(config.LNG_RANGE), "land": [0, 10_000], "description": 5_000},
        "category_labels": {c: CATEGORY_LABELS[c] for c in model.categories_},
        "gazetteer": model.gazetteer_,
        "ranges": model.ranges_,
        "interval": list(model.interval_),
        "threshold": float(model.threshold_),
        "options": model.options(),
        "card": card,
        "forest": {"n_trees": len(offsets), "tree_offsets": offsets, "total_nodes": total_nodes},
        "svm": {"n_features": int(rff.W_.shape[0]), "n_components": int(rff.W_.shape[1]),
                "intercept": float(lin.intercept_)},
        "binary": streams,
        "density": density,
    }


def export(model: PricePredictor, out: Path, data: Path | None = None, standalone: bool = False) -> dict:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    feat, thr, val, offsets = flatten_forest(model)
    core, explain = encode_forest(feat, thr, val, len(model.pre_.feature_names_))
    rff, lin = model.svm_.rff_, model.svm_.svm_
    core += [("W", rff.W_), ("c", rff.c_), ("coef", lin.coef_)]
    streams = {"core": write_stream(out, "model", core), "explain": write_stream(out, "explain", explain)}
    meta = build_meta(model, streams, offsets, len(feat), density_grid(model, data))
    (out / "model.json").write_text(json.dumps(meta, allow_nan=False, separators=(",", ":")))
    (out / "index.html").write_text(build_page(standalone))
    return meta


def write_stream(out: Path, stem: str, sections: list[tuple[str, np.ndarray]]) -> dict:
    """
    Pack, gzip and split one set of arrays into <stem>-<k>.txt files, base64
    encoded: static hosts (and claude.ai artifacts) serve text everywhere but
    not always arbitrary binary types, and most compress text in transit.
    """
    raw, layout = pack_sections(sections)
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    files, text_bytes = [], 0
    for k in range(0, len(packed), CHUNK_BYTES):
        files.append(f"{stem}-{len(files)}.txt")
        text = base64.b64encode(packed[k:k + CHUNK_BYTES])
        (out / files[-1]).write_bytes(text)
        text_bytes += len(text)
    return {"encoding": "gzip+base64", "files": files, "bytes": len(packed), "text_bytes": text_bytes,
            "raw_bytes": len(raw), "sections": layout}


def build_page(standalone: bool = False) -> str:
    """The page with its two scripts inlined (one file, no build tools)."""
    page = (HERE / "index.html").read_text()
    for name in ("predictor.js", "ui.js"):
        marker = f'<script src="{name}"></script>'
        if marker not in page:
            raise ValueError(f"{marker} missing from index.html")
        page = page.replace(marker, f"<script>\n{(HERE / name).read_text()}</script>")
    if standalone:     # a full document for hosts that don't wrap the page (GitHub Pages, Netlify)
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                f"</head>\n<body>\n{page}\n</body>\n</html>\n")
    return page


# ----------------------------------------------------------------- JS vs Python
PHRASES = ["Metroya yaxın", "dənizə baxır", "avro təmirli", "Təcili satılır", "əşyalı", "qaraj var",
           "Срочно продается", "евроремонт", "вид на море", "паркинг", "Mebelli", "Sea view flat",
           "Ev çox işıqlıdır", "kupça hazırdır", "İpoteka mümkündür", "Şəhərin mərkəzində"]


def random_listings(model: PricePredictor, n: int, seed: int = 0) -> list[dict]:
    """Valid listings that exercise every input path (missing values, map pins,
    houses with land, unknown-length descriptions with Azerbaijani/Russian text)."""
    rnd = random.Random(seed)
    locs = list(model.gazetteer_)
    (la0, la1), (lo0, lo1) = MAP_BOX["lat"], MAP_BOX["lng"]
    maybe = lambda p, v: v if rnd.random() < p else None  # noqa: E731
    out = []
    for _ in range(n):
        cat = rnd.choice(model.categories_)
        house = cat == "heyet evi/bag evi"
        total = maybe(0.85, rnd.randint(1, 30))
        floor = maybe(0.85, rnd.randint(-1, total) if total else rnd.randint(1, 20))
        if floor is not None and total is not None and floor > total:
            floor = total
        pin = rnd.random() < 0.3
        words = rnd.sample(PHRASES, rnd.randint(0, 4))
        item = {
            "category": cat,
            "area_m2": round(math.exp(rnd.uniform(math.log(20), math.log(600))), rnd.choice([0, 0, 1])),
            "rooms": maybe(0.9, rnd.randint(1, 8)),
            "floor": None if house else floor, "total_floors": None if house else total,
            "land_area_sot": maybe(0.8, round(rnd.uniform(1, 20), 1)) if house else maybe(0.1, 3.0),
            "location": maybe(0.8, rnd.choice(locs)),
            "lat": round(rnd.uniform(la0, la1), 5) if pin else None,
            "lng": round(rnd.uniform(lo0, lo1), 5) if pin else None,
            "repair": rnd.choice([True, False, None]),
            "mortgage": rnd.random() < 0.2, "bill_of_sale": rnd.random() < 0.6,
            "description": " ".join(words) + (" " * rnd.randint(0, 3)) + ("x" * rnd.randint(0, 300)),
        }
        out.append(item)
    return out


def golden(model: PricePredictor, items: list[dict]) -> list[dict]:
    return [{"input": it, "expected": asdict(model.predict(Listing(**it)))} for it in items]


def check_js(model: PricePredictor, site: Path, n: int, seed: int = 0) -> dict:
    """Run predictor.js under Node on ``n`` random listings and compare with Python."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node is not installed; cannot check the JS port")
    cases = site.parent / f"{site.name}-golden.json"
    cases.write_text(json.dumps(golden(model, random_listings(model, n, seed))))
    res = subprocess.run([node, str(HERE / "check.js"), str(site), str(cases)],
                         capture_output=True, text=True, timeout=900)
    if res.returncode not in (0, 1):
        raise RuntimeError(res.stderr or res.stdout)
    return json.loads(res.stdout)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="bundle written by app.train")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (replaced)")
    ap.add_argument("--data", help="dataset for the map's density layer (default: data/house_sale.csv if present)")
    ap.add_argument("--standalone", action="store_true", help="wrap index.html in a full HTML document")
    ap.add_argument("--check", type=int, default=0, metavar="N", help="compare JS vs Python on N listings")
    args = ap.parse_args(argv)

    model = PricePredictor.load(args.model)
    data = None
    try:
        data = dp.find_data_file(args.data)
    except FileNotFoundError:
        print("No dataset found: the map will show locations only (no density layer).")
    if data is not None and model.meta_.get("data_sha256") not in (None, _sha256(data)):
        print(f"{data} is not the file the bundle was trained on: skipping the density layer.")
        data = None
    meta = export(model, Path(args.out), data, args.standalone)
    sizes = {p.name: p.stat().st_size for p in sorted(Path(args.out).iterdir())}
    print(f"Wrote {args.out}: " + ", ".join(f"{k} {v / 2**20:.1f} MB" for k, v in sizes.items()))
    b = meta["binary"]
    print(f"  {meta['forest']['total_nodes']:,} tree nodes; download {b['core']['text_bytes'] / 2**20:.1f} MB for "
          f"estimates + {b['explain']['text_bytes'] / 2**20:.1f} MB for explanations")
    if args.check:
        res = check_js(model, Path(args.out), args.check)
        print(json.dumps(res, indent=1))
        if res["failures"]:
            sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
