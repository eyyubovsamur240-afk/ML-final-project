"""
Train the bundle the web interface serves.

    python -m app.train                       # real data, report settings (~15 min)
    python -m app.train --fast                # subsample + small models (~20 s, for trying the UI)
    python -m app.train --data other.csv --out models/other.pkl
"""

from __future__ import annotations

import argparse
import sys

from src import config
from src import data_prep as dp

from .predictor import DEFAULT_MODEL_PATH, PricePredictor, _sha256


def main(argv=None) -> PricePredictor:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to the bina.az CSV (default: data/house_sale.csv)")
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH), help="where to write the bundle")
    ap.add_argument("--fast", action="store_true", help="subsample and small models (smoke test)")
    args = ap.parse_args(argv)

    path = dp.find_data_file(args.data)
    print(f"Loading {path}")
    cleaned = dp.clean(dp.load_raw(path))
    if args.fast and len(cleaned) > config.FAST["max_rows"]:
        cleaned = cleaned.sample(n=config.FAST["max_rows"], random_state=config.SEED).reset_index(drop=True)
    print(f"{len(cleaned):,} cleaned residential listings")

    model = PricePredictor().fit(cleaned, fast=args.fast)
    model.meta_["data_file"] = path.name
    model.meta_["data_sha256"] = _sha256(path)
    out = model.save(args.out)

    card = model.card()
    print(f"Held-out test: price RMSE(log) {card['test_rmse_log']:.3f}, R² {card['test_r2_log']:.3f}; "
          f"tier F1 {card['test_f1']:.3f}, ROC-AUC {card['test_roc_auc']:.3f}")
    print(f"80% price range covers {card['interval_coverage']:.0%} of test listings")
    print(f"Saved {out}")
    return model


if __name__ == "__main__":
    main(sys.argv[1:])
