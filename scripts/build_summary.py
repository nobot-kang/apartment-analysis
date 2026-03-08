"""집계 파일만 재생성하는 스크립트.

Raw 데이터가 이미 수집된 상태에서 집계 로직만 다시 실행할 때 사용한다.

Usage::

    python scripts/build_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipelines.aggregation_pipeline import AggregationPipeline


def main() -> None:
    """집계 파이프라인을 실행한다."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level="INFO",
    )

    logger.info("집계 파이프라인 시작")
    pipeline = AggregationPipeline()
    pipeline.run_all()
    logger.info("집계 파이프라인 완료")


if __name__ == "__main__":
    main()
