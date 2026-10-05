"""
data_prep.py — loading, cleaning, feature building, splitting, tier label.

ALL data wrangling lives here so the rest of the code only calls these
functions. The pipeline is split in two halves on purpose:

  * Row-wise steps that use NO statistics of the data (parsing strings,
    currency conversion with fixed rates, fixed domain rules for typos,
    de-duplication, engineered columns). These are safe to run on the full
    dump before splitting.
  * Steps that LEARN something (median imputation, which category levels get
    their own one-hot column, standardisation, the price-tier threshold).
    These live in ``Preprocessor`` / ``Standardizer`` / ``make_tier_label``
    and are fitted on the TRAINING rows only, then applied to val/test.

The raw dump is scraped and messy: column names are Azerbaijani, numbers
arrive as strings ("85 m²", "1 250 000", "5 / 9"), prices come in several
currencies. The parsers below are deliberately tolerant and every row that
gets dropped is counted in a cleaning log that goes into the report.
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# --- Config -----------------------------------------------------------------
SEED = config.SEED
DATA_PATH = config.DATA_PATH

# Columns that LEAK the target because they are arithmetic functions of price
# (price per m², price restated). They are dropped before anything else.
LEAKAGE_COLUMNS: list[str] = ["unit_price", "total_price"]

# Identifiers / personal data / free-form address: no predictive meaning we
# are allowed to use, and they would let a model memorise individual sellers.
IDENTIFIER_COLUMNS: list[str] = [
    "id", "item_id", "url", "link", "owner_name", "owner_title", "owner",
    "shop_name", "shop_title", "shop", "address", "phone", "contact",
]

# Listing-promotion metadata: describes the seller's ad budget, not the
# property, and its meaning changes with bina.az pricing policy.
LISTING_META_COLUMNS: list[str] = [
    "vip", "featured", "products_label", "extra_info", "updated", "created",
    "date", "views",
]

# Canonical (English) name -> accepted spellings after ``normalize_name``.
COLUMN_ALIASES: dict[str, list[str]] = {
    "price": ["price", "qiymet"],
    "currency": ["currency", "valyuta"],
    "area_raw": ["sahe", "area", "area_m2", "sahe_m2"],
    "rooms_raw": ["otaq_sayi", "rooms", "room_count", "otaq"],
    "floor_raw": ["mertebe", "floor"],
    "land_raw": ["torpaq_sahesi", "land_area", "land"],
    "lat": ["lat", "latitude"],
    "lng": ["lng", "lon", "long", "longitude"],
    "location": ["location", "district", "rayon", "qesebe"],
    "city": ["city", "seher"],
    "building_type": ["binanin_novu", "building_type"],
    "category": ["kateqoriya", "category"],
    "repair_raw": ["temir", "repair"],
    "mortgage_raw": ["ipoteka", "mortgage"],
    "bill_of_sale_raw": ["cixaris", "bill_of_sale", "kupca"],
    "description": ["description", "tesvir", "melumat"],
    "attributes": ["attributes", "attrs"],
}

# Final feature lists produced by ``make_features``.
NUMERIC_FEATURES = [
    "area_m2", "log_area", "rooms", "area_per_room", "floor", "total_floors",
    "floor_ratio", "is_top_floor", "is_first_floor", "land_area_sot", "has_land",
    "lat", "lng", "dist_center_km", "desc_log_len",
]
BINARY_FEATURES = [
    "repair", "mortgage", "bill_of_sale",
    "kw_metro", "kw_sea", "kw_furnished", "kw_urgent", "kw_euro_reno", "kw_parking",
]
CATEGORICAL_FEATURES = ["category", "building_type", "city", "location"]
# Missing-indicator columns are added only for these base features (derived
# columns such as floor_ratio are missing exactly when their base is).
INDICATOR_FEATURES = ["rooms", "floor", "total_floors", "lat", "repair"]

# Keyword flags searched in the transliterated description (no numbers are
# read from the text, so prices written inside a description cannot leak).
DESCRIPTION_KEYWORDS = {
    "kw_metro": ["metro", "метро"],
    "kw_sea": ["deniz", "море", "sea view"],
    "kw_furnished": ["mebel", "esyali", "мебел"],
    "kw_urgent": ["tecili", "срочно"],
    "kw_euro_reno": ["avro temir", "euro temir", "евроремонт", "avrotemir"],
    "kw_parking": ["qaraj", "parking", "dayanacaq", "паркинг"],
}

_AZ_TRANSLIT = str.maketrans({
    "ə": "e", "Ə": "e", "ı": "i", "İ": "i", "ö": "o", "Ö": "o", "ü": "u",
    "Ü": "u", "ç": "c", "Ç": "c", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
})


# =============================================================================
# String helpers
# =============================================================================
def transliterate(text: str) -> str:
    """Lower-case and strip Azerbaijani diacritics: 'Otaq sayı' -> 'otaq sayi'."""
    text = str(text).translate(_AZ_TRANSLIT)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def normalize_name(name: str) -> str:
    """Column name -> snake_case ASCII: 'Binanın növü' -> 'binanin_novu'."""
    return re.sub(r"[^a-z0-9]+", "_", transliterate(name)).strip("_")


_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(value) -> float:
    """
    Tolerant number parser for scraped strings.
      '85 m²' -> 85, '1 250 000' -> 1250000, '85,5' -> 85.5,
      '1,250,000' -> 1250000, None/'' -> nan.
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = re.sub(r"[\s  ]", "", str(value))
    if not s:
        return np.nan
    if _THOUSANDS.match(s):                 # 1,250,000
        s = s.replace(",", "")
    elif "," in s and "." not in s:         # 85,5 (decimal comma)
        s = s.replace(",", ".")
    m = _NUMBER.search(s)
    return float(m.group()) if m else np.nan


def parse_floor(value) -> tuple[float, float]:
    """'5 / 9' -> (5, 9);  '5' -> (5, nan);  junk -> (nan, nan)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, np.nan
    nums = re.findall(r"-?\d+", str(value))
    if not nums:
        return np.nan, np.nan
    floor = float(nums[0])
    total = float(nums[1]) if len(nums) > 1 else np.nan
    return floor, total


def parse_land_sot(value) -> float:
    """Land area in 'sot' (1 sot = 100 m²): '6 sot' -> 6, '600 m²' -> 6, '0.5 ha' -> 50."""
    x = parse_number(value)
    if np.isnan(x):
        return np.nan
    s = transliterate(value) if isinstance(value, str) else ""
    if "ha" in s.split() or s.endswith("ha"):
        return x * 100.0
    if "m2" in s or "m²" in str(value) or "kv" in s:
        return x / 100.0
    return x


def parse_area_m2(value) -> float:
    """Floor area in m². Some land listings state 'sot' -> convert to m²."""
    x = parse_number(value)
    if isinstance(value, str) and "sot" in transliterate(value):
        return x * 100.0
    return x


_YES = {"var", "beli", "he", "yes", "true", "1", "movcuddur", "bəli", "есть", "да"}
_NO = {"yox", "yoxdur", "xeyr", "no", "false", "0", "нет"}


def parse_yes_no(value) -> float:
    """'var' -> 1, 'yoxdur' -> 0, missing/unknown -> nan."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    s = transliterate(value).strip()
    if s in _YES:
        return 1.0
    if s in _NO:
        return 0.0
    return np.nan


def parse_currency(value, price_text=None) -> str:
    """Normalise a currency marker to 'AZN'/'USD'/'EUR'/'RUB' (default AZN)."""
    for text in (value, price_text):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            continue
        s = transliterate(text)
        if "usd" in s or "$" in s or "dollar" in s:
            return "USD"
        if "eur" in s or "€" in s or "avro" in s:
            return "EUR"
        if "rub" in s or "₽" in s:
            return "RUB"
        if "azn" in s or "₼" in s or "manat" in s:
            return "AZN"
    return "AZN"


def _parse_attributes(value) -> dict:
    """Best-effort parse of an 'attributes' blob (JSON, Python dict, or 'k: v; k: v')."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    s = str(value).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(s)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):        # [{"name": k, "value": v}, ...]
            out = {}
            for item in obj:
                if isinstance(item, dict) and len(item) >= 2:
                    k = item.get("name", item.get("key"))
                    if k is not None:
                        out[k] = item.get("value")
            return out
    out = {}
    for part in re.split(r"[;\n|]", s):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


# =============================================================================
# Loading
# =============================================================================
def find_data_file(path: str | Path | None = None) -> Path:
    """Use ``path`` if it exists, else DATA_PATH, else the only CSV in data/."""
    candidates = [Path(path)] if path else []
    candidates.append(Path(DATA_PATH))
    for p in candidates:
        if p.exists():
            return p
    csvs = sorted(config.DATA_DIR.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    raise FileNotFoundError(
        f"Dataset not found at {candidates[0]}. Download the bina.az sale dataset "
        "from Kaggle (see data/README.md) and put the CSV in data/, or pass "
        "--data path/to/file.csv."
    )


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw CSV (UTF-8, all columns as strings) and return it unmodified."""
    path = find_data_file(path)
    return pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=True,
                       low_memory=False)


# =============================================================================
# Cleaning
# =============================================================================
@dataclass
class CleaningLog:
    """Row counts after every cleaning step (goes into the report table)."""
    steps: list[tuple[str, int]] = field(default_factory=list)
    dropped_columns: dict[str, list[str]] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)

    def add(self, step: str, n_rows: int) -> None:
        self.steps.append((step, int(n_rows)))

    def as_dict(self) -> dict:
        return {"steps": self.steps, "dropped_columns": self.dropped_columns,
                "notes": self.notes}


def canonicalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename raw (Azerbaijani/English) columns to canonical English names."""
    norm = {c: normalize_name(c) for c in df.columns}
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for raw, n in norm.items():
            if n in aliases and raw not in rename:
                rename[raw] = canon
                break
    out = df.rename(columns=rename)
    # keep un-mapped columns under their normalised names
    out = out.rename(columns={c: norm[c] for c in out.columns if c in norm and c not in rename})
    return out, rename


def _fill_from_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """If key columns only live inside an 'attributes' blob, pull them out."""
    needed = [c for c in ("area_raw", "rooms_raw", "floor_raw", "category",
                          "building_type", "repair_raw", "mortgage_raw",
                          "bill_of_sale_raw", "land_raw") if c not in df.columns]
    if not needed or "attributes" not in df.columns:
        return df
    parsed = df["attributes"].map(_parse_attributes)
    lookup = {a: canon for canon, aliases in COLUMN_ALIASES.items() for a in aliases}
    extracted: dict[str, list] = {c: [None] * len(df) for c in needed}
    for i, attrs in enumerate(parsed):
        for k, v in attrs.items():
            canon = lookup.get(normalize_name(k))
            if canon in extracted:
                extracted[canon][i] = v
    df = df.copy()
    for c, values in extracted.items():
        if any(v is not None for v in values):
            df[c] = values
    return df


def _is_price_derived(col: str) -> bool:
    return col != "price" and ("price" in col or "qiymet" in col)


def clean(df: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    """
    Clean the raw frame (row-wise rules only — nothing is learned here):
      1. canonical column names; drop leakage, identifier and listing-meta columns,
      2. parse numbers / floors / land / yes-no strings, convert price to AZN,
      3. drop rows without a usable price or area,
      4. drop exact and near duplicates (re-posted listings),
      5. drop implausible rows by fixed domain rules (typos, unit mix-ups),
      6. set impossible feature values (rooms, floors, coordinates) to missing;
         they are imputed later from TRAIN statistics.
    """
    log = log if log is not None else CleaningLog()
    log.add("raw rows", len(df))
    df, rename = canonicalize_columns(df)
    log.notes["column_mapping"] = rename
    df = _fill_from_attributes(df)
    if "price" not in df.columns:
        raise KeyError(f"No price column found. Columns: {list(df.columns)}")

    # --- 1. leakage / identifiers / meta --------------------------------------
    leak = [c for c in df.columns if c in LEAKAGE_COLUMNS or _is_price_derived(c)]
    ids = [c for c in df.columns if c in IDENTIFIER_COLUMNS]
    meta = [c for c in df.columns if c in LISTING_META_COLUMNS]
    log.dropped_columns = {"leakage": leak, "identifiers": ids, "listing_meta": meta}
    df = df.drop(columns=leak + ids + meta)

    # --- 2. parse ---------------------------------------------------------------
    out = pd.DataFrame(index=df.index)
    currency = [parse_currency(c, p) for c, p in
                zip(df.get("currency", pd.Series([None] * len(df), index=df.index)),
                    df["price"])]
    out["currency"] = currency
    rate = pd.Series(currency, index=df.index).map(config.TO_AZN).astype(float)
    out["price_azn"] = df["price"].map(parse_number) * rate
    log.notes["currency_counts"] = pd.Series(currency).value_counts().to_dict()

    out["area_m2"] = df["area_raw"].map(parse_area_m2) if "area_raw" in df else np.nan
    out["rooms"] = df["rooms_raw"].map(parse_number) if "rooms_raw" in df else np.nan
    if "floor_raw" in df:
        fl = df["floor_raw"].map(parse_floor)
        out["floor"] = [f for f, _ in fl]
        out["total_floors"] = [t for _, t in fl]
    else:
        out["floor"] = out["total_floors"] = np.nan
    if "total_floors" in df:        # some dumps have a separate column
        out["total_floors"] = out["total_floors"].fillna(df["total_floors"].map(parse_number))
    out["land_area_sot"] = df["land_raw"].map(parse_land_sot) if "land_raw" in df else np.nan
    for c in ("lat", "lng"):
        out[c] = df[c].map(parse_number) if c in df else np.nan
    for src, dst in (("repair_raw", "repair"), ("mortgage_raw", "mortgage"),
                     ("bill_of_sale_raw", "bill_of_sale")):
        out[dst] = df[src].map(parse_yes_no) if src in df else np.nan
    for c in ("category", "building_type", "city", "location"):
        if c in df:
            s = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            out[c] = s.map(lambda v: transliterate(v).strip() if isinstance(v, str) and v else np.nan)
        else:
            out[c] = np.nan
    out["description"] = df["description"].fillna("") if "description" in df else ""

    # --- 3. target & core feature must exist -----------------------------------
    out = out[out["price_azn"].notna() & (out["price_azn"] > 0)]
    log.add("has a positive price", len(out))
    out = out[out["area_m2"].notna() & (out["area_m2"] > 0)]
    log.add("has a positive area", len(out))

    # --- 4. duplicates ------------------------------------------------------------
    out = out.drop_duplicates()
    log.add("exact duplicates removed", len(out))
    key = ["price_azn", "area_m2", "rooms", "floor", "total_floors", "lat", "lng", "category"]
    out = out.drop_duplicates(subset=key)
    log.add("re-posted listings removed", len(out))

    # --- 5. fixed plausibility rules (typos / non-sale listings) -------------------
    lo, hi = config.PRICE_RANGE_AZN
    out = out[out["price_azn"].between(lo, hi)]
    log.add(f"price in [{lo:,}, {hi:,}] AZN", len(out))
    lo, hi = config.AREA_RANGE_M2
    out = out[out["area_m2"].between(lo, hi)]
    log.add(f"area in [{lo}, {hi}] m2", len(out))
    ppm = out["price_azn"] / out["area_m2"]
    lo, hi = config.PRICE_PER_M2_RANGE
    out = out[ppm.between(lo, hi)]
    log.add(f"price/m2 in [{lo}, {hi:,}] (typo filter only)", len(out))

    # --- 6. impossible feature values -> missing ------------------------------------
    lo, hi = config.ROOMS_RANGE
    out.loc[~out["rooms"].between(lo, hi), "rooms"] = np.nan
    bad_total = (out["total_floors"] < 1) | (out["total_floors"] > config.MAX_TOTAL_FLOORS)
    out.loc[bad_total, "total_floors"] = np.nan
    bad_floor = (out["floor"] < -2) | (out["floor"] > out["total_floors"].fillna(np.inf))
    out.loc[bad_floor, "floor"] = np.nan
    out.loc[out["land_area_sot"] < 0, "land_area_sot"] = np.nan
    bad_geo = ~(out["lat"].between(*config.LAT_RANGE) & out["lng"].between(*config.LNG_RANGE))
    out.loc[bad_geo, ["lat", "lng"]] = np.nan
    log.notes["missing_after_cleaning"] = out.isna().mean().round(4).to_dict()
    log.add("final cleaned rows", len(out))
    return out.reset_index(drop=True)


# =============================================================================
# Features
# =============================================================================
def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km (vectorised)."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Engineer row-wise features from the cleaned frame.

    Returns (features, y, feature_columns) where ``features`` still holds NaNs
    and raw category strings: imputation and one-hot encoding LEARN from data,
    so they happen in ``Preprocessor`` after the split (no val/test leakage).
    ``y`` is the price in AZN.
    """
    f = pd.DataFrame(index=df.index)
    f["area_m2"] = df["area_m2"]
    f["log_area"] = np.log(df["area_m2"])
    f["rooms"] = df["rooms"]
    f["area_per_room"] = df["area_m2"] / df["rooms"]
    f["floor"] = df["floor"]
    f["total_floors"] = df["total_floors"]
    f["floor_ratio"] = df["floor"] / df["total_floors"]
    f["is_top_floor"] = np.where(df["total_floors"].notna() & df["floor"].notna(),
                                 (df["floor"] == df["total_floors"]).astype(float), np.nan)
    f["is_first_floor"] = np.where(df["floor"].notna(), (df["floor"] <= 1).astype(float), np.nan)
    f["land_area_sot"] = df["land_area_sot"].fillna(0.0)   # apartments have no land
    f["has_land"] = (f["land_area_sot"] > 0).astype(float)
    f["lat"] = df["lat"]
    f["lng"] = df["lng"]
    f["dist_center_km"] = haversine_km(df["lat"], df["lng"], *config.BAKU_CENTER)
    desc = df["description"].fillna("").astype(str)
    f["desc_log_len"] = np.log1p(desc.str.len())
    f["repair"] = df["repair"]
    # bina.az only shows the mortgage / bill-of-sale fields when they apply,
    # so an absent field means "no", not "unknown".
    f["mortgage"] = df["mortgage"].fillna(0.0)
    f["bill_of_sale"] = df["bill_of_sale"].fillna(0.0)
    desc_t = desc.map(transliterate)
    for name, words in DESCRIPTION_KEYWORDS.items():
        f[name] = desc_t.map(lambda s, ws=words: float(any(w in s for w in ws)))
    for c in CATEGORICAL_FEATURES:
        f[c] = df[c]
    cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    return f[cols], df["price_azn"].to_numpy(dtype=float), cols


class Preprocessor:
    """
    Turns the engineered frame into a numeric design matrix.

    Learned on TRAIN only (``fit``), applied unchanged to val/test:
      * numeric/binary: median imputation + a missing-indicator column for
        each base feature in ``indicators`` that had missing values in TRAIN,
      * categoricals: one-hot for levels with >= ``min_count`` TRAIN rows
        (top ``max_levels``); everything else -> "<col>=other". One-hot (not
        ordinal) because district/category have no natural order; trees can
        still isolate a single level with one split.
    """

    def __init__(self, numeric=None, binary=None, categorical=None, indicators=None,
                 min_count: int = config.MIN_CATEGORY_COUNT,
                 max_levels: int = config.MAX_CATEGORY_LEVELS):
        self.numeric = list(numeric if numeric is not None else NUMERIC_FEATURES)
        self.binary = list(binary if binary is not None else BINARY_FEATURES)
        self.categorical = list(categorical if categorical is not None else CATEGORICAL_FEATURES)
        self.indicators = list(indicators if indicators is not None else INDICATOR_FEATURES)
        self.min_count = min_count
        self.max_levels = max_levels

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        cont = self.numeric + self.binary
        values = df[cont].astype(float)
        self.medians_ = values.median().fillna(0.0).to_dict()
        self.indicator_cols_ = [c for c in self.indicators
                                if c in values and values[c].isna().any()]
        self.levels_ = {}
        for c in self.categorical:
            counts = df[c].dropna().value_counts()
            counts = counts[counts >= self.min_count].head(self.max_levels)
            self.levels_[c] = sorted(counts.index.tolist())
        self.feature_names_ = (
            cont
            + [f"{c}_missing" for c in self.indicator_cols_]
            + [f"{c}={lvl}" for c in self.categorical for lvl in self.levels_[c]]
            + [f"{c}=other" for c in self.categorical]
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cont = self.numeric + self.binary
        values = df[cont].astype(float)
        blocks = [values.fillna(self.medians_).to_numpy()]
        if self.indicator_cols_:
            blocks.append(values[self.indicator_cols_].isna().to_numpy(dtype=float))
        onehots, others = [], []
        for c in self.categorical:
            col = df[c].to_numpy(dtype=object)
            lv = self.levels_[c]
            m = np.zeros((len(df), len(lv)))
            for j, level in enumerate(lv):
                m[:, j] = col == level
            onehots.append(m)
            others.append((m.sum(axis=1) == 0).astype(float)[:, None])
        blocks.extend(onehots)
        blocks.extend(others)
        return np.hstack(blocks).astype(float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


# =============================================================================
# Target, split, scaling
# =============================================================================
def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER.
      premium = 1 (price > threshold), standard = 0.
    Call it with the TRAINING prices and ``threshold=None`` to get the training
    median; then pass that threshold when labelling val/test (no peeking).
    Returns (y_tier, threshold).
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), float(threshold)


def price_strata(y_price, n_bins: int = config.N_STRATA) -> np.ndarray:
    """Quantile bin of log(price), used ONLY to allocate rows to splits."""
    logp = np.log(np.asarray(y_price, dtype=float))
    edges = np.quantile(logp, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, logp, side="right")


def split_indices(n: int, val_size=config.VAL_SIZE, test_size=config.TEST_SIZE,
                  seed=SEED, stratify=None):
    """
    Deterministic (stratified) train/val/test index split.
    Each stratum is shuffled with a seeded RNG and cut in the same proportions,
    so every split sees the same price distribution (and ~50/50 tier balance).
    """
    rng = np.random.default_rng(seed)
    strata = np.zeros(n, dtype=int) if stratify is None else np.asarray(stratify)
    tr, va, te = [], [], []
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        n_te = int(round(test_size * len(idx)))
        n_va = int(round(val_size * len(idx)))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    return tuple(np.sort(np.concatenate(p)) for p in (tr, va, te))


def train_val_test_split(X, y, val_size=config.VAL_SIZE, test_size=config.TEST_SIZE,
                         seed=SEED, stratify=None):
    """Return (X_tr, y_tr, X_val, y_val, X_te, y_te); works for arrays and DataFrames."""
    tr, va, te = split_indices(len(y), val_size, test_size, seed, stratify)

    def take(a, idx):
        return a.iloc[idx] if isinstance(a, (pd.DataFrame, pd.Series)) else np.asarray(a)[idx]
    return take(X, tr), take(y, tr), take(X, va), take(y, va), take(X, te), take(y, te)


def stratified_kfold(strata, k: int = config.CV_FOLDS, seed: int = SEED):
    """Yield (train_idx, test_idx) for k stratified folds (deterministic)."""
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(strata), dtype=int)
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        fold_of[idx] = np.arange(len(idx)) % k
    for f in range(k):
        yield np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)


class Standardizer:
    """z-scoring with TRAIN mean/std (constant columns keep std = 1)."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def standardize(X_tr, *others):
    """
    Standardize using TRAIN statistics only, then apply the same transform to
    the other splits. Returns the scaled arrays in the same order as given.
    """
    sc = Standardizer().fit(X_tr)
    out = [sc.transform(X_tr)] + [sc.transform(X) for X in others]
    return tuple(out) if others else out[0]
