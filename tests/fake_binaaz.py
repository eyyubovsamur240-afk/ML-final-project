"""
fake_binaaz.py — a SYNTHETIC stand-in for the bina.az dump, for tests only.

It reproduces the *format* problems of the real scrape (Azerbaijani column
names, numbers stored as strings with units, mixed currencies, leakage
columns, identifiers, duplicates, typos, missing values) so the whole
pipeline can be exercised in CI without the Kaggle file. Prices follow a
made-up generative model, so NUMBERS PRODUCED FROM THIS FILE MEAN NOTHING —
never put them in the report.

    python -m tests.fake_binaaz /tmp/fake.csv 6000
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

DISTRICTS = {   # name: (lat, lng, price multiplier)
    "Nəsimi r.": (40.385, 49.835, 1.6), "Yasamal r.": (40.385, 49.810, 1.4),
    "Nərimanov r.": (40.405, 49.870, 1.35), "Səbail r.": (40.360, 49.835, 1.9),
    "Xətai r.": (40.385, 49.950, 1.1), "Binəqədi r.": (40.460, 49.830, 0.85),
    "Nizami r.": (40.400, 49.950, 0.95), "Sabunçu r.": (40.440, 49.950, 0.7),
    "Suraxanı r.": (40.420, 50.040, 0.6), "Xəzər r.": (40.470, 50.150, 0.65),
    "Qaradağ r.": (40.250, 49.600, 0.55), "28 May m.": (40.380, 49.850, 1.7),
    "Gənclik m.": (40.400, 49.850, 1.5), "Badamdar q.": (40.340, 49.800, 1.3),
}
CITIES = {"Bakı": 0.86, "Sumqayıt": 0.06, "Gəncə": 0.04, "Xırdalan": 0.04}
CITY_CENTRES = {"Sumqayıt": (40.59, 49.67), "Gəncə": (40.68, 46.36), "Xırdalan": (40.45, 49.75)}
CATEGORIES = {"Yeni tikili": 0.45, "Köhnə tikili": 0.30, "Həyət evi / Bağ evi": 0.18,
              "Ofis": 0.05, "Obyekt": 0.02}
CAT_MULT = {"Yeni tikili": 1.15, "Köhnə tikili": 0.95, "Həyət evi / Bağ evi": 0.8,
            "Ofis": 1.3, "Obyekt": 1.1}
PHRASES = ["Təcili satılır", "Metroya yaxın", "Dəniz mənzərəli", "Əşyalı satılır",
           "Avro təmirli", "Qaraj var", "Kupça var", "Sənədləri qaydasındadır"]


def make_fake_binaaz(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    city = rng.choice(list(CITIES), n, p=list(CITIES.values()))
    cat = rng.choice(list(CATEGORIES), n, p=list(CATEGORIES.values()))
    dist_names = list(DISTRICTS)
    dist = rng.choice(dist_names, n)
    lat = np.empty(n)
    lng = np.empty(n)
    mult = np.empty(n)
    for i in range(n):
        if city[i] == "Bakı":
            la, ln, m = DISTRICTS[dist[i]]
        else:
            (la, ln), m = CITY_CENTRES[city[i]], 0.5
            dist[i] = city[i] + " ş."
        lat[i], lng[i], mult[i] = la + rng.normal(0, 0.012), ln + rng.normal(0, 0.012), m

    house = cat == "Həyət evi / Bağ evi"
    rooms = np.clip(rng.poisson(2.3, n) + 1 + house, 1, 9)
    area = np.round(rooms * rng.uniform(22, 38, n) + rng.normal(0, 8, n) + 40 * house).clip(15)
    total_floors = np.where(house, rng.integers(1, 3, n), rng.integers(4, 26, n))
    floor = np.array([rng.integers(1, t + 1) for t in total_floors])
    repair = rng.random(n) < 0.7
    mortgage = rng.random(n) < 0.3
    bill = rng.random(n) < 0.75
    land = np.where(house, np.round(rng.uniform(1, 12, n), 1), np.nan)
    desc_flags = rng.random((n, len(PHRASES))) < 0.15
    desc = [". ".join(p for p, f in zip(PHRASES, row) if f) or "Mənzil satılır" for row in desc_flags]
    seaview = desc_flags[:, 2]

    per_m2 = (1300 * mult * np.vectorize(CAT_MULT.get)(cat)
              * (1 + 0.25 * repair) * (1 + 0.15 * seaview) * (1 + 0.02 * np.minimum(floor, 15))
              * np.exp(rng.normal(0, 0.18, n)))
    price = np.round(per_m2 * area + np.nan_to_num(land) * 4000, -2)

    usd = rng.random(n) < 0.08
    shown_price = np.where(usd, np.round(price / 1.7, -2), price)
    df = pd.DataFrame({
        "price": [f"{int(p):,}".replace(",", " ") for p in shown_price],
        "currency": np.where(usd, "USD", "AZN"),
        "unit_price": [f"{int(u):,} AZN/m²".replace(",", " ") for u in price / area],
        "total_price": price,
        "location": dist,
        "city": city,
        "Kateqoriya": cat,
        "Binanın növü": np.where(cat == "Yeni tikili", "Yeni tikili",
                                 np.where(cat == "Köhnə tikili", "Köhnə tikili", None)),
        "Sahə": [f"{a:g} m²" for a in area],
        "Otaq sayı": rooms.astype(str),
        "Mərtəbə": [f"{f} / {t}" for f, t in zip(floor, total_floors)],
        "Torpaq sahəsi": [f"{x:g} sot" if not np.isnan(x) else None for x in land],
        "Təmir": np.where(repair, "var", "yoxdur"),
        "İpoteka": np.where(mortgage, "var", None),
        "Çıxarış": np.where(bill, "var", None),
        "lat": np.round(lat, 6).astype(str),
        "lng": np.round(lng, 6).astype(str),
        "description": desc,
        "owner_name": [f"Owner {i}" for i in rng.integers(0, 2000, n)],
        "address": [f"Street {i}" for i in rng.integers(0, 5000, n)],
        "vip": rng.random(n) < 0.1,
    })
    # --- realistic dirt -----------------------------------------------------
    miss = rng.random(n)
    df.loc[miss < 0.04, "Otaq sayı"] = None
    df.loc[(miss > 0.04) & (miss < 0.07), ["lat", "lng"]] = None
    df.loc[(miss > 0.07) & (miss < 0.075), "price"] = None
    df.loc[(miss > 0.075) & (miss < 0.08), "price"] = "1"              # placeholder price
    df.loc[(miss > 0.08) & (miss < 0.085), "Sahə"] = "8500 m²"          # typo
    df.loc[(miss > 0.085) & (miss < 0.09), "Mərtəbə"] = "12 / 5"        # floor > total
    dupes = df.sample(frac=0.05, random_state=seed)
    return pd.concat([df, dupes], ignore_index=True).sample(frac=1.0, random_state=seed)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "fake_binaaz.csv"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    make_fake_binaaz(n).to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out} (SYNTHETIC, for testing only)")
