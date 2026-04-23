"""cleaned_apt_trade 데이터셋 독립 빌드 스크립트.

Usage::

    # canonical 빌드 (연도 청크 + 통합본 + summary)
    uv run python scripts/build_cleaned_trade.py

    # 통합본 생략
    uv run python scripts/build_cleaned_trade.py --no-combined

    # 특정 연도만
    uv run python scripts/build_cleaned_trade.py --years 2023 2024

    # variant: A-3 진단 컬럼 포함 (variants/ 에만 저장)
    uv run python scripts/build_cleaned_trade.py --variant with-diagnostics

    # variant: 1층 제외
    uv run python scripts/build_cleaned_trade.py --variant exclude-floor1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from loguru import logger
from pipelines.cleaned_trade_pipeline import build_cleaned_trade


def main() -> None:
    parser = argparse.ArgumentParser(description="cleaned_apt_trade 데이터셋 빌드")
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="통합본(cleaned_apt_trade.parquet) 생성 생략",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        metavar="YYYY",
        help="처리할 연도 목록 (기본: 전체)",
    )
    parser.add_argument(
        "--variant",
        choices=["with-diagnostics", "exclude-floor1"],
        default=None,
        help="variant 산출물 생성 (canonical 파일은 건드리지 않음)",
    )
    args = parser.parse_args()

    try:
        build_cleaned_trade(
            include_combined=not args.no_combined,
            years=args.years,
            variant=args.variant,
        )
    except Exception as exc:
        logger.error(f"빌드 실패: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
