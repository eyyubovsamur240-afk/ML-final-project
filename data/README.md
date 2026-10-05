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
**directly in this folder**: `data/house_sale.csv` (the file inside the archive).

The loader (`src/data_prep.py::find_data_file`) uses `config.DATA_PATH`
(`data/house_sale.csv`) if it exists, otherwise **the only `*.csv` in `data/`**.
If you have several CSVs, either set `DATA_PATH` once in `src/config.py` or run
`python -m src.run_all --data data/<file>.csv`.

## Provenance (the copy our results were produced from)
| Field | Value |
|-------|-------|
| Downloaded on | 2026-10-05 (Kaggle API, `sehriyarmemmedli/binaaz-sale-project`) |
| File name | `house_sale.csv` (141.9 MB, 100,775 rows, 51 columns) |
| Scrape period | 2024-10-05 → 2024-11-19 |
| SHA-256 | `b6a4c67e0d3c712de31b9db53db53688792bc03099a213c1994c43114758aec2` |

`run_all` re-computes the hash into `results/metrics.json` → `data.sha256`; if yours differs,
the Kaggle file has changed and numbers may differ slightly from the report.

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
- Prices in USD/EUR are converted to AZN with the fixed rates in `config.TO_AZN`
  (in this copy every listing is already in AZN).
- The scraper visited many listings several times (100,775 rows, 64,454 distinct listing URLs);
  the cleaner keeps the latest scrape of each listing.
- Only residential categories are kept (`config.KEEP_CATEGORIES`); set it to `None` to keep land,
  commercial objects, offices and garages too.
