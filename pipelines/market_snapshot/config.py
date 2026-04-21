"""Section A 파이프라인 공통 상수 및 파라미터."""

from __future__ import annotations

from pathlib import Path

import sys

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 출력 디렉토리
PREPROCESSED_PLUS_DIR: Path = _project_root / "data" / "preprocessed_plus"

# 면적 구간 정의 (㎡)
AREA_BUCKETS = [
    (0,   60,   "~60㎡"),
    (60,  85,   "60~85㎡"),
    (85,  102,  "85~102㎡"),
    (102, 9999, "102㎡~"),
]

# 이상치 탐지: moving average band 의 최소 상대 폭
OUTLIER_THRESHOLD: float = 0.25
# 시세 조회 최대 소급 개월 수
LOOKBACK_MONTHS: int = 6
# Bollinger band 파라미터
BOLLINGER_WINDOW_MONTHS: int = 6
BOLLINGER_MIN_HISTORY_MONTHS: int = 3
BOLLINGER_STD_MULTIPLIER: float = 2.0
# 급격한 가격 이동이 추세 전환인지 확인하는 파라미터
TREND_LOOKAHEAD_MONTHS: int = 6
TREND_MIN_SUPPORT_MONTHS: int = 2
TREND_MIN_TOTAL_TRADES: int = 3
TREND_SUPPORT_BAND_RATIO: float = 0.5
TREND_ALIGNMENT_TOLERANCE: float = 0.12
# 추세 전환으로 인정된 월 안에서 개별 행을 다시 점검할 때 쓰는 band
TREND_ROW_MIN_TRADE_COUNT: int = 3
TREND_ROW_STD_MULTIPLIER: float = 2.5
TREND_ROW_MIN_BAND_PCT: float = 0.08

# A-3 v2: spread-based outlier detection
PATH_WINDOW_MONTHS: int = 7
PATH_MIN_PERIODS: int = 3
SPREAD_WINDOW_MONTHS: int = 9
SHRINK_K: int = 6
BAND_Z: float = 3.0
FLOOR_PCT_BASE: float = 0.18
FLOOR_PCT_SPARSE_ADDON: float = 0.05
SANITY_LOG_RATIO: float = 0.693          # ln(2.0)
LEADER_SPREAD_MONTHS: int = 24
LEADER_SPREAD_SIGN_RATIO: float = 0.80
OLD_COMPLEX_AGE: int = 20
RENOVATION_ABS_BUFFER_MANWON: int = 5_000   # 5천만 원 = 5,000 만원
RENOVATION_REL_CAP: float = 0.12
ABS_DEVIATION_MANWON: int = 30_000          # 3억 원 = 30,000 만원
