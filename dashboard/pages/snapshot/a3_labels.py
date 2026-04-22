"""A-3 이상치 섹션 — 판정사유·단지유형 레이블 상수."""

A3_REASON_LABELS: dict[str, str] = {
    "unsupported_jump": "지지 없는 단발 점프",
    "sanity_error": "입력/단위 오류",
    "trend_month_robust_band": "추세월 내부 스파이크",
    "abs_deviation": "절대 금액 이탈 (±3억)",
    "legacy_band_outlier": "legacy 밴드 이상치",
    "uncategorized": "미분류",
}

A3_REASON_ORDER = [
    "unsupported_jump",
    "sanity_error",
    "trend_month_robust_band",
    "abs_deviation",
    "legacy_band_outlier",
    "uncategorized",
]

A3_STRUCTURE_LABELS: dict[str, str] = {
    "normal": "일반 단지",
    "leader_or_isolated": "대장/나홀로 단지",
    "legacy_unknown": "미분류 (legacy 데이터)",
    "unknown": "미분류",
}

A3_STRUCTURE_ORDER = [
    "normal",
    "leader_or_isolated",
    "legacy_unknown",
    "unknown",
]
