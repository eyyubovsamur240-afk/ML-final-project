# Data

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`):
it is third-party data containing listing details, and the project rules forbid
redistributing it. Each team member downloads it locally.

## Where to get it
Kaggle: **"binaaz-sale-project"** by `sehriyarmemmedli`
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

## How to place it
Either with the Kaggle CLI (needs `~/.kaggle/kaggle.json`):
```bash
pip install kaggle
kaggle datasets download -d sehriyarmemmedli/binaaz-sale-project -p data --unzip
```
or download the archive in the browser and extract it so the CSV lives
**directly in this folder**, e.g. `data/bina_az_sale.csv`.

The loader (`src/data_prep.py::find_data_file`) uses `config.DATA_PATH`
(`data/bina_az_sale.csv`) if it exists, otherwise **the only `*.csv` in `data/`**.
If you have several CSVs, either set `DATA_PATH` once in `src/config.py` or run
`python -m src.run_all --data data/<file>.csv`.

## Provenance (fill in once)
| Field | Value |
|-------|-------|
| Downloaded on | YYYY-MM-DD |
| Kaggle version | vN |
| File name | … |
| SHA-256 | printed by `run_all` into `results/metrics.json` → `data.sha256` |

## Notes / gotchas handled by the code
- Column names are **Azerbaijani** (`Sahə`, `Otaq sayı`, `Mərtəbə`, `Torpaq sahəsi`,
  `Binanın növü`, `Kateqoriya`, `Təmir`, `İpoteka`, `Çıxarış`, …). The file is read as UTF-8 and
  names are transliterated/mapped in `data_prep.COLUMN_ALIASES`. If your copy names a column
  differently, add the spelling there; nothing else needs to change.
- If those fields only exist inside an `attributes` blob (JSON / dict / `k: v; …`), they are
  extracted automatically.
- **Leakage:** `unit_price`, `total_price` and any other column whose name contains `price` are dropped
  (`data_prep.LEAKAGE_COLUMNS`); identifiers/addresses and promotion flags are dropped too.
  The exact list is printed in `results/metrics.json` → `data.dropped_columns` and in the report.
- Prices in USD/EUR are converted to AZN with the fixed rates in `config.TO_AZN`.
