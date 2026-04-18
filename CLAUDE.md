# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

Seoul/Gyeonggi apartment real-estate analysis dashboard. Ingests MOLIT transactions, ECOS macro indicators, and yfinance market data; aggregates to parquet; served by a multi-page Streamlit app.

## Commands

```bash
uv sync                                              # base deps (Python 3.11)
uv sync --extra advanced                             # + sklearn/geopandas/hmmlearn

uv run python scripts/run_full_pipeline.py           # full collection + aggregation
#   flags: --mode retry | --trade-only | --rent-only | --skip-molit | --building-ledger-only
uv run python scripts/build_summary.py               # rebuild aggregates only

uv run streamlit run streamlit_app.py                # dashboard
uv run pytest                                        # tests
```

## Environment

Copy `.env.example` → `.env`:
- `MOLIT_API_KEY` (data.go.kr)
- `ECOS_API_KEY` (ecos.bok.or.kr)

## References

- Architecture, data flow, key modules, page map → [docs/architecture.md](docs/architecture.md)
- Original project/analysis plans → `apartment_analysis_plan.md`, `real_estate_project_plan.md`, `real_estate_implementation_guide.md`, `real_estate_complex_info_analysis_plan.md`, `real_estate_representative_complex_analysis_plan.md`
