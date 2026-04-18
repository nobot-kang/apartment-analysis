# Architecture

## Data Flow

```
Raw collection (pipelines/)
    molit_pipeline.py             → data/raw/molit/apt_trade/{region}_{YYYYMM}.parquet
                                  → data/raw/molit/apt_rent/{region}_{YYYYMM}.parquet
    ecos_pipeline.py              → data/raw/ecos/
    market_pipeline.py            → data/raw/market/
    market_snapshot_pipeline.py   → data/raw/market_snapshot/
    building_ledger_pipeline.py   → data/raw/building_ledger/
    apartment_list.py             → apartment complex universe helpers
        ↓
Preprocessing (pipelines/data_preprocessing.py)
    DataPreprocessor.preprocess_trade() / preprocess_rent()
    → filters cancelled transactions, normalizes columns, computes derived features
        ↓
Aggregation (pipelines/aggregation_pipeline.py + building_ledger_summary.py)
    → data/processed/*.parquet  (monthly summaries, complex panels, macro data)
        ↓
Dashboard (dashboard/)
    data_loader.py   wraps analysis/common.py loaders with @st.cache_resource
    app.py           NAVIGATION dict → lazy-loads page render functions
    pages/page_NN_*.py   render_*() functions called by app.py
```

## Key Modules

- **`config/settings.py`** — single source of truth for constants: API endpoints, region codes (`SEOUL_REGIONS`, `GYEONGGI_REGIONS`, `ALL_REGIONS`), date range (`START_YM`/`END_YM`), ECOS stat codes, yfinance tickers. `get_api_key()` reads `st.secrets` first, then `os.getenv`.

- **`analysis/common.py`** — shared data loaders (`load_trade_summary_df`, `load_complex_monthly_panel_df`, etc.) and constants (area bins, floor bins, policy event dates, district coordinates).

- **`analysis/level{1-4}.py`** — analysis functions by complexity level (1=basic overview, 4=advanced forecasting/ML). Dashboard pages call these directly.

- **`analysis/complex_analysis.py`** / **`analysis/representative_complex_analysis.py`** — complex-level analysis (hedonic regression, panel FE, spillover, forecasting).

- **`dashboard/app.py`** — `NAVIGATION` dict maps sidebar labels → `(module, function)`; modules imported lazily via `importlib.import_module`. Pattern for adding pages.

- **`pipelines/representative_complex_pipeline.py`** — derives the "representative complex" universe (59㎡ and 84㎡ type units) used in pages 10–13.

- **`pipelines/market_snapshot_pipeline.py`** — market snapshot collection; diagnostics on page 00.

## Dashboard Pages

- **00** — market snapshot diagnostics
- **01–05** — district-level analysis (Level 1–4)
- **06–09** — individual complex analysis (Complex Level 1–4)
- **10–13** — representative complex analysis (Representative Level 1–4)
- **14** — trade filter diagnostics

Each page module exposes multiple `render_*()` functions registered in `dashboard/app.py:NAVIGATION`.

## Data Storage

- `data/raw/` — git-ignored, per-region per-month parquet files
- `data/processed/` — committed, aggregated parquets consumed by the dashboard
- Raw MOLIT filename convention: `{5-digit-region-code}_{YYYYMM}.parquet`

## CI

`.github/workflows/update_data.yml` runs on the 1st of each month: incremental collection, rebuild summaries, commit `data/processed/`.
