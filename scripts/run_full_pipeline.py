"""전체 파이프라인 일괄 실행 스크립트.

국토부 실거래가, ECOS 거시지표, yfinance 시장 데이터를 순차 수집하고
집계 파이프라인을 실행한다.

Usage::

    # 전체 수집 (증분 모드 – 이미 있는 파일 skip)
    python scripts/run_full_pipeline.py

    # 증분 모드 명시
    python scripts/run_full_pipeline.py --mode incremental

    # 실패 항목 재시도
    python scripts/run_full_pipeline.py --mode retry

    # 매매만 수집
    python scripts/run_full_pipeline.py --trade-only

    # 전월세만 수집
    python scripts/run_full_pipeline.py --rent-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import (
    END_DATE,
    END_YM,
    START_DATE,
    START_YM,
    get_api_key,
)
from pipelines.aggregation_pipeline import AggregationPipeline
from pipelines.ecos_pipeline import EcosPipeline
from pipelines.market_pipeline import MarketPipeline
from pipelines.molit_pipeline import MolitPipeline


def _configure_logger() -> None:
    """loguru 로거 설정."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
    )
    log_path = _project_root / "logs"
    log_path.mkdir(exist_ok=True)
    logger.add(
        log_path / "pipeline_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
    )


def run_molit(
    mode: str = "incremental",
    trade_only: bool = False,
    rent_only: bool = False,
) -> None:
    """국토부 실거래가 수집을 실행한다.

    Args:
        mode: ``incremental`` 또는 ``retry``.
        trade_only: 매매만 수집 여부.
        rent_only: 전월세만 수집 여부.
    """
    api_key = get_api_key("MOLIT_API_KEY")
    if not api_key:
        logger.error("MOLIT_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return

    pipeline = MolitPipeline(api_key=api_key)

    if mode == "retry":
        pipeline.retry_failed()
    else:
        collect_trade = not rent_only
        collect_rent = not trade_only
        pipeline.run_full_collection(
            start_ym=START_YM,
            end_ym=END_YM,
            collect_trade=collect_trade,
            collect_rent=collect_rent,
        )


def run_ecos() -> None:
    """ECOS 거시지표 수집을 실행한다."""
    api_key = get_api_key("ECOS_API_KEY")
    if not api_key:
        logger.error("ECOS_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return

    pipeline = EcosPipeline(api_key=api_key)
    pipeline.run_all(start_ym=START_YM, end_ym=END_YM)


def run_market() -> None:
    """yfinance 시장 데이터 수집을 실행한다."""
    pipeline = MarketPipeline()
    pipeline.run_all(start=START_DATE, end=END_DATE)


def run_aggregation() -> None:
    """집계 파이프라인을 실행한다."""
    pipeline = AggregationPipeline()
    pipeline.run_all()


def main() -> None:
    """메인 진입점."""
    parser = argparse.ArgumentParser(description="부동산 실거래가 전체 파이프라인")
    parser.add_argument(
        "--mode",
        choices=["incremental", "retry"],
        default="incremental",
        help="실행 모드 (기본: incremental)",
    )
    parser.add_argument(
        "--trade-only",
        action="store_true",
        help="매매 데이터만 수집",
    )
    parser.add_argument(
        "--rent-only",
        action="store_true",
        help="전월세 데이터만 수집",
    )
    parser.add_argument(
        "--skip-molit",
        action="store_true",
        help="국토부 수집 건너뛰기",
    )
    parser.add_argument(
        "--skip-ecos",
        action="store_true",
        help="ECOS 수집 건너뛰기",
    )
    parser.add_argument(
        "--skip-market",
        action="store_true",
        help="yfinance 수집 건너뛰기",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="집계 건너뛰기",
    )
    args = parser.parse_args()

    _configure_logger()
    logger.info("=== 전체 파이프라인 시작 ===")

    # 1. 국토부 실거래가
    if not args.skip_molit:
        logger.info("--- 국토부 실거래가 수집 ---")
        run_molit(
            mode=args.mode,
            trade_only=args.trade_only,
            rent_only=args.rent_only,
        )

    # 2. ECOS 거시지표
    if not args.skip_ecos:
        logger.info("--- ECOS 거시지표 수집 ---")
        run_ecos()

    # 3. yfinance 시장 데이터
    if not args.skip_market:
        logger.info("--- yfinance 시장 데이터 수집 ---")
        run_market()

    # 4. 집계
    if not args.skip_aggregation:
        logger.info("--- 집계 파이프라인 ---")
        run_aggregation()

    logger.info("=== 전체 파이프라인 완료 ===")


if __name__ == "__main__":
    main()
