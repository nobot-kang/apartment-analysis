"""A-3 이상치 섹션 — 순수 필터·전처리 헬퍼 (UI 프레임워크 의존 없음)."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from dashboard.pages.snapshot.a3_labels import (
    A3_REASON_LABELS,
    A3_REASON_ORDER,
    A3_STRUCTURE_LABELS,
    A3_STRUCTURE_ORDER,
)

__all__ = [
    "_resolve_color_col",
    "_ordered_present_keys",
    "_prepare_a3_filter_frame",
]


def _resolve_color_col(
    mode: str,
    reason_key: str | None,
    available_columns: Iterable[str],
) -> tuple[str, bool]:
    """그래프 색상 컬럼과 폴백 여부를 반환한다.

    Returns:
        (color_col, fell_back_from_direction)
        fell_back_from_direction=True 이면 caller 가 st.info 경고를 1회 표시해야 한다.
    """
    cols = set(available_columns)

    if mode == "reason":
        return "판정사유", False

    wants_direction = mode == "direction" or (mode == "auto" and reason_key is not None)
    if wants_direction:
        if "outlier_direction" in cols:
            return "outlier_direction", False
        return "판정사유", True  # old schema 폴백

    # mode == "auto" and reason_key is None
    return "판정사유", False


def _ordered_present_keys(series: pd.Series, preferred_order: list[str]) -> list[str]:
    """선호 순서를 우선한 뒤 나머지 값을 정렬해 반환한다."""
    values = [str(v) for v in series.dropna().unique().tolist() if str(v).strip()]
    present = set(values)
    ordered = [key for key in preferred_order if key in present]
    extras = sorted(present - set(preferred_order))
    return ordered + extras


def _prepare_a3_filter_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, bool]:
    """A-3 필터/표시용 레이블 컬럼을 준비한다."""
    prepared = df.copy()

    has_reason_schema = (
        "outlier_reason" in prepared.columns
        and prepared["outlier_reason"].dropna().astype(str).str.strip().ne("").any()
    )
    if has_reason_schema:
        reason_key = prepared["outlier_reason"].fillna("uncategorized").astype(str)
        reason_key = reason_key.where(reason_key.str.strip().ne(""), "uncategorized")
    elif "reference_type" in prepared.columns and prepared["reference_type"].notna().any():
        reason_key = (
            prepared["reference_type"]
            .map(
                {
                    "moving_average_band": "legacy_band_outlier",
                    "trend_month_robust_band": "trend_month_robust_band",
                    "sanity_error": "sanity_error",
                }
            )
            .fillna("legacy_band_outlier")
            .astype(str)
        )
    else:
        reason_key = pd.Series("uncategorized", index=prepared.index, dtype="object")
    prepared["_reason_filter_key"] = reason_key
    prepared["판정사유"] = prepared["_reason_filter_key"].map(A3_REASON_LABELS).fillna(prepared["_reason_filter_key"])

    has_structure_schema = (
        "structure_type" in prepared.columns
        and prepared["structure_type"].dropna().astype(str).str.strip().ne("").any()
    )
    if has_structure_schema:
        structure_key = prepared["structure_type"].fillna("unknown").astype(str)
        structure_key = structure_key.where(structure_key.str.strip().ne(""), "unknown")
    else:
        structure_key = pd.Series("legacy_unknown", index=prepared.index, dtype="object")
    prepared["_structure_filter_key"] = structure_key
    prepared["단지유형"] = prepared["_structure_filter_key"].map(A3_STRUCTURE_LABELS).fillna(prepared["_structure_filter_key"])

    return prepared, has_reason_schema, has_structure_schema
