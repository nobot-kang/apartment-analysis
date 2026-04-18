"""Section A 파이프라인 오케스트레이션."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines.market_snapshot.config import PREPROCESSED_PLUS_DIR
from pipelines.market_snapshot.io import _load_all_trade, _load_all_rent
from pipelines.market_snapshot.preprocess import (
    _add_region_columns,
    _add_area_bucket,
    _add_month_column,
)
from pipelines.market_snapshot.snapshot_monthly import (
    build_snapshot_monthly_trade,
    build_snapshot_monthly_rent,
)
from pipelines.market_snapshot.snapshot_area_mix import build_snapshot_area_mix
from pipelines.market_snapshot.outliers import build_snapshot_outliers

import sys
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import PROCESSED_DIR


class MarketSnapshotPipeline:
    """Section A 시장 스냅샷 집계 파이프라인."""

    def __init__(
        self,
        processed_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.output_dir = output_dir or PREPROCESSED_PLUS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, df: pd.DataFrame, filename: str) -> None:
        """parquet으로 저장하고 파일 크기를 확인한다."""
        if df.empty:
            logger.warning(f"저장 스킵 (빈 DataFrame): {filename}")
            return
        out_path = self.output_dir / filename
        df.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info(f"저장 완료: {filename} ({len(df):,}행, {size_mb:.1f} MB)")
        if size_mb > 90:
            logger.warning(f"파일 크기 주의: {filename} ({size_mb:.1f} MB) – GitHub LFS 한도(100MB) 근접")

    def run(self) -> None:
        """전체 파이프라인을 실행한다."""
        logger.info("=== MarketSnapshotPipeline 시작 ===")

        trade_raw = _load_all_trade(self.processed_dir)
        rent_raw = _load_all_rent(self.processed_dir)

        if trade_raw.empty:
            logger.error("매매 데이터가 없습니다. 파이프라인을 중단합니다.")
            return

        logger.info(f"매매 로드 완료: {len(trade_raw):,}건")
        logger.info(f"전월세 로드 완료: {len(rent_raw):,}건")

        logger.info("공통 전처리 (region, area_bucket, month 컬럼 추가)...")
        trade_df = _add_region_columns(trade_raw)
        trade_df = _add_area_bucket(trade_df)
        trade_df = _add_month_column(trade_df)

        if not rent_raw.empty:
            rent_df = _add_region_columns(rent_raw)
            rent_df = _add_area_bucket(rent_df)
            rent_df = _add_month_column(rent_df)
        else:
            rent_df = pd.DataFrame()

        monthly_trade = build_snapshot_monthly_trade(trade_df)
        self._save(monthly_trade, "snapshot_monthly_trade.parquet")

        if not rent_df.empty:
            monthly_rent = build_snapshot_monthly_rent(rent_df)
            self._save(monthly_rent, "snapshot_monthly_rent.parquet")

        area_mix = build_snapshot_area_mix(trade_df)
        self._save(area_mix, "snapshot_area_mix.parquet")

        outliers, market_price = build_snapshot_outliers(trade_df)
        self._save(outliers, "snapshot_outliers.parquet")
        self._save(market_price, "snapshot_complex_market_price.parquet")

        logger.info("=== MarketSnapshotPipeline 완료 ===")
