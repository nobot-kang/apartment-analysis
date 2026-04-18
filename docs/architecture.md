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

- **`pipelines/market_snapshot_pipeline.py`** — 호환 shim. 실제 구현은 `pipelines/market_snapshot/` 패키지에 있다. 직접 실행 진입점(`uv run python pipelines/market_snapshot_pipeline.py`)과 기존 import 경로를 유지한다.

- **`pipelines/market_snapshot/`** — Section A 집계 패키지.
  - `config.py` — 파이프라인 전용 상수 (Bollinger 파라미터, spread 파라미터 등)
  - `io.py` — `_load_all_trade` / `_load_all_rent`
  - `preprocess.py` — region / area_bucket / month 컬럼 파생
  - `snapshot_monthly.py` — A-1 매매·전월세 월별 집계
  - `snapshot_area_mix.py` — A-2 면적 믹스 & 구성효과 분해
  - `outliers/` — A-3 이상치 탐지 서브패키지
    - `_smoothing.py` — 공용 로그 rolling-median helper
    - `cohort_paths.py` — C1/C2 코호트 가격 경로
    - `complex_spreads.py` — G0/G1/G2 spread + shrinkage 기준가
    - `dynamic_band.py` — sigma_eff → band_pct 계산
    - `trend_band.py` — legacy Bollinger band + 추세 전환 확인
    - `pipeline.py` — `build_snapshot_outliers` 최종 조합 엔트리
  - `runner.py` — `MarketSnapshotPipeline` 오케스트레이션

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
