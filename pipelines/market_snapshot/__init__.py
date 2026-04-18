"""pipelines.market_snapshot 패키지 공개 API."""

from __future__ import annotations

from pipelines.market_snapshot.runner import MarketSnapshotPipeline
from pipelines.market_snapshot.snapshot_monthly import (
    build_snapshot_monthly_trade,
    build_snapshot_monthly_rent,
)
from pipelines.market_snapshot.snapshot_area_mix import build_snapshot_area_mix
from pipelines.market_snapshot.outliers import build_snapshot_outliers

# private helper 재노출 (테스트 호환)
from pipelines.market_snapshot.preprocess import _add_area_bucket, _add_region_columns
from pipelines.market_snapshot.outliers.cohort_paths import _build_cohort_paths
from pipelines.market_snapshot.outliers.complex_spreads import _build_complex_spreads
from pipelines.market_snapshot.outliers.dynamic_band import _compute_dynamic_band

__all__ = [
    "MarketSnapshotPipeline",
    "build_snapshot_monthly_trade",
    "build_snapshot_monthly_rent",
    "build_snapshot_area_mix",
    "build_snapshot_outliers",
    "_add_area_bucket",
    "_add_region_columns",
    "_build_cohort_paths",
    "_build_complex_spreads",
    "_compute_dynamic_band",
]
