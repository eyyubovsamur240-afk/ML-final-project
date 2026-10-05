"""
config.py — every constant, path, rule and hyperparameter grid in ONE place.

Nothing in here is learned from data. Changing a value here and re-running
``python -m src.run_all`` is the only supported way to change an experiment.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility ----------------------------------------------------------
SEED = 42

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "bina_az_sale.csv"      # change if your filename differs
RESULTS_DIR = ROOT / "results"                  # json/csv outputs (git-ignored)
FIGURES_DIR = ROOT / "report" / "figures"       # every figure (git-ignored)
GENERATED_TEX_DIR = ROOT / "report" / "generated"  # tables/macros for LaTeX

# --- Currency -----------------------------------------------------------------
# Fixed conversion to AZN. The manat has been pegged at 1.70 AZN/USD by the
# Central Bank of Azerbaijan since 2017; EUR uses an approximate 2025-26 rate.
# Rates are constants (not estimated from the data) so they cannot leak.
TO_AZN = {"AZN": 1.0, "USD": 1.70, "EUR": 1.97, "RUB": 0.02}

# --- Cleaning rules (fixed domain rules, documented in the report) ------------
PRICE_RANGE_AZN = (5_000, 15_000_000)     # outside: data-entry error / not a sale
AREA_RANGE_M2 = (10, 3_000)               # outside: typo or unit mix-up
PRICE_PER_M2_RANGE = (30, 25_000)         # used ONLY to filter typos, never a feature
ROOMS_RANGE = (1, 20)                     # outside -> set missing (imputed later)
MAX_TOTAL_FLOORS = 60                     # tallest Baku towers are ~40-50 floors
LAT_RANGE = (38.3, 41.95)                 # Azerbaijan bounding box
LNG_RANGE = (44.7, 50.9)
BAKU_CENTER = (40.3777, 49.8920)          # Fountain Square, for distance feature
BAKU_BBOX = {"lat": (40.25, 40.65), "lng": (49.60, 50.40)}   # map zoom (Absheron)

# --- Feature encoding -----------------------------------------------------------
MIN_CATEGORY_COUNT = 20     # a level needs >= this many TRAIN rows to get its own column
MAX_CATEGORY_LEVELS = 60    # cap per categorical column; rarer levels -> "other"

# --- Split ----------------------------------------------------------------------
VAL_SIZE = 0.15
TEST_SIZE = 0.15
N_STRATA = 10               # stratify on deciles of log(price) -> also balances the tier
CV_FOLDS = 5

# --- Hyperparameter grids (selection uses the VALIDATION split only) -----------
TREE_DEPTHS = list(range(1, 26))
TREE_MIN_LEAF_GRID = [1, 2, 5, 10, 20, 50, 100, 200]
TREE_MIN_DECREASE_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
LEARNING_CURVE_FRACTIONS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

SVM_LAMBDAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SVM_EPOCHS = 50
SVM_BATCH = 32
SVM_AVERAGE = True          # t-weighted iterate averaging (see src/svm.py)
RFF_GAMMAS = [0.003, 0.01, 0.03, 0.1]          # RBF bandwidth k(x,z)=exp(-g||x-z||^2)
RFF_LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3]
RFF_COMPONENTS = 1024
RFF_COMPONENT_GRID = [32, 64, 128, 256, 512, 1024, 2048]

RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

FOREST_TREES = 60
BOOST_TREES = 200
BOOST_LR = 0.1
BOOST_DEPTH = 4

KMEANS_K_GRID = list(range(2, 21))
KMEANS_K = 12

# Smaller settings for `python -m src.run_all --fast` (smoke test, CI).
FAST = {
    "max_rows": 4_000,
    "tree_depths": list(range(1, 13)),
    "svm_epochs": 8,
    "svm_lambdas": [1e-5, 1e-4, 1e-3, 1e-2],
    "rff_gammas": [0.01, 0.1],
    "rff_lambdas": [1e-5, 1e-4],
    "rff_components": 256,
    "rff_component_grid": [32, 128, 256],
    "forest_trees": 8,
    "boost_trees": 25,
    "cv_folds": 3,
    "n_boot": 100,
}
N_BOOT = 1_000              # bootstrap resamples for test-set confidence intervals
