"""Section A – 데이터 진단 & 시장 스냅샷 집계 파이프라인 (호환 shim).

실제 구현은 pipelines/market_snapshot/ 패키지에 있습니다.
이 파일은 기존 import 경로 및 직접 실행 경로 호환을 위해 유지합니다.

출력 위치: data/preprocessed_plus/
  - snapshot_monthly_trade.parquet          : A-1 월별 거래량·가격 추이 (매매)
  - snapshot_monthly_rent.parquet           : A-1 월별 거래량·가격 추이 (전월세)
  - snapshot_area_mix.parquet               : A-2 면적 믹스 변화 & 구성효과 분해
  - snapshot_outliers.parquet               : A-3 이상치·비정상 거래 탐지 결과
  - snapshot_complex_market_price.parquet   : A-3 단지별 월별 시세 (이상치 제외)
"""

from __future__ import annotations

from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Public API re-exports (기존 import 경로 호환)
# ---------------------------------------------------------------------------
from pipelines.market_snapshot.runner import MarketSnapshotPipeline  # noqa: F401
from pipelines.market_snapshot.snapshot_monthly import (  # noqa: F401
    build_snapshot_monthly_trade,
    build_snapshot_monthly_rent,
)
from pipelines.market_snapshot.snapshot_area_mix import build_snapshot_area_mix  # noqa: F401
from pipelines.market_snapshot.outliers import build_snapshot_outliers  # noqa: F401

# Private helper re-exports (테스트 호환)
from pipelines.market_snapshot.preprocess import _add_area_bucket, _add_region_columns  # noqa: F401
from pipelines.market_snapshot.outliers.cohort_paths import _build_cohort_paths  # noqa: F401
from pipelines.market_snapshot.outliers.complex_spreads import _build_complex_spreads  # noqa: F401
from pipelines.market_snapshot.outliers.dynamic_band import _compute_dynamic_band  # noqa: F401

# Constants re-exports
from pipelines.market_snapshot.config import (  # noqa: F401
    PREPROCESSED_PLUS_DIR,
    AREA_BUCKETS,
    OUTLIER_THRESHOLD,
    LOOKBACK_MONTHS,
    BOLLINGER_WINDOW_MONTHS,
    BOLLINGER_MIN_HISTORY_MONTHS,
    BOLLINGER_STD_MULTIPLIER,
    TREND_LOOKAHEAD_MONTHS,
    TREND_MIN_SUPPORT_MONTHS,
    TREND_MIN_TOTAL_TRADES,
    TREND_SUPPORT_BAND_RATIO,
    TREND_ALIGNMENT_TOLERANCE,
    TREND_ROW_MIN_TRADE_COUNT,
    TREND_ROW_STD_MULTIPLIER,
    TREND_ROW_MIN_BAND_PCT,
    PATH_WINDOW_MONTHS,
    PATH_MIN_PERIODS,
    SPREAD_WINDOW_MONTHS,
    SHRINK_K,
    BAND_Z,
    FLOOR_PCT_BASE,
    FLOOR_PCT_SPARSE_ADDON,
    SANITY_LOG_RATIO,
    LEADER_SPREAD_MONTHS,
    LEADER_SPREAD_SIGN_RATIO,
    OLD_COMPLEX_AGE,
    RENOVATION_ABS_BUFFER_KRW,
    RENOVATION_REL_CAP,
)


if __name__ == "__main__":
    MarketSnapshotPipeline().run()
