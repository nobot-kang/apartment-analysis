# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seoul/Gyeonggi apartment real-estate analysis dashboard. Ingests transaction data from the Korean Ministry of Land (MOLIT) API, macro indicators from Bank of Korea (ECOS), and market data via yfinance. Aggregates everything into parquet files served by a multi-page Streamlit dashboard.

## Commands

```bash
# Install dependencies (Python 3.11 required)
uv sync                        # base dependencies
uv sync --extra advanced       # includes scikit-learn, geopandas, hmmlearn, etc.

# Data collection (requires API keys in .env)
uv run python scripts/run_full_pipeline.py                   # full incremental collection + aggregation
uv run python scripts/run_full_pipeline.py --mode retry      # retry failed requests
uv run python scripts/run_full_pipeline.py --trade-only      # only apartment trades
uv run python scripts/run_full_pipeline.py --rent-only       # only rent transactions
uv run python scripts/run_full_pipeline.py --skip-molit      # skip MOLIT collection
uv run python scripts/run_full_pipeline.py --building-ledger-only

# Rebuild aggregated parquets only (raw data already collected)
uv run python scripts/build_summary.py

# Run the dashboard
uv run streamlit run streamlit_app.py

# Tests
uv run pytest
uv run pytest tests/test_specific.py  # single test file
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:
- `MOLIT_API_KEY` — from data.go.kr (국토부 실거래가 API)
- `ECOS_API_KEY` — from ecos.bok.or.kr (한국은행 ECOS API)

When running inside Streamlit, `config/settings.py:get_api_key()` reads from `st.secrets` first, then falls back to `os.getenv`.

## Architecture

### Data Flow

```
Raw collection (pipelines/)
    molit_pipeline.py        → data/raw/molit/apt_trade/{region}_{YYYYMM}.parquet
                             → data/raw/molit/apt_rent/{region}_{YYYYMM}.parquet
    ecos_pipeline.py         → data/raw/ecos/
    market_pipeline.py       → data/raw/market/
    building_ledger_pipeline.py → data/raw/building_ledger/
        ↓
Preprocessing (pipelines/data_preprocessing.py)
    DataPreprocessor.preprocess_trade() / preprocess_rent()
    → filters cancelled transactions, normalizes columns, computes derived features
        ↓
Aggregation (pipelines/aggregation_pipeline.py + building_ledger_summary.py)
    → data/processed/*.parquet  (monthly summaries, complex panels, macro data)
        ↓
Dashboard (dashboard/)
    data_loader.py           wraps analysis/common.py loaders with @st.cache_resource
    app.py                   NAVIGATION dict → lazy-loads page render functions
    pages/page_NN_*.py       render_*() functions called by app.py
```

### Key Modules

- **`config/settings.py`** — single source of truth for all constants: API endpoints, region codes (`SEOUL_REGIONS`, `GYEONGGI_REGIONS`, `ALL_REGIONS`), date range (`START_YM`/`END_YM`), ECOS stat codes, yfinance tickers.

- **`analysis/common.py`** — shared data loaders (`load_trade_summary_df`, `load_complex_monthly_panel_df`, etc.) and constants used across all analysis and dashboard modules (area bins, floor bins, policy event dates, district coordinates).

- **`analysis/level{1-4}.py`** — analysis functions grouped by complexity level (1=basic overview, 4=advanced forecasting/ML). Dashboard pages call these directly.

- **`analysis/complex_analysis.py`** / **`analysis/representative_complex_analysis.py`** — apartment complex-level analysis (hedonic regression, panel FE, spillover effects, forecasting).

- **`dashboard/app.py`** — the NAVIGATION dict maps sidebar labels to `(module, function)` tuples; modules are imported lazily via `importlib.import_module`. This is the pattern for adding new pages.

- **`pipelines/representative_complex_pipeline.py`** — derives the "representative complex" universe (59㎡ and 84㎡ type units) used in pages 10–13.

### Dashboard Page Structure

Pages follow levels:
- Pages 01–05: district-level analysis (Level 1–4)
- Pages 06–09: individual complex analysis (Complex Level 1–4)
- Pages 10–13: representative complex analysis (Representative Level 1–4)

Each page module exposes multiple `render_*()` functions registered in `dashboard/app.py:NAVIGATION`.

### Data Storage

- `data/raw/` — git-ignored, per-region per-month parquet files
- `data/processed/` — committed to git, aggregated parquet files consumed by the dashboard
- Raw MOLIT data filename convention: `{5-digit-region-code}_{YYYYMM}.parquet`

### CI

GitHub Actions (`.github/workflows/update_data.yml`) runs on the 1st of each month: collects incremental data, rebuilds summaries, and commits `data/processed/` back to the repo.
