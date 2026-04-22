"""Unit tests for _ordered_present_keys and _prepare_a3_filter_frame (hermetic)."""

import pandas as pd
import pytest

from dashboard.pages.snapshot.a3_filters import (
    _ordered_present_keys,
    _prepare_a3_filter_frame,
)


# ---------------------------------------------------------------------------
# _ordered_present_keys
# ---------------------------------------------------------------------------

def test_ordered_present_keys_preferred_first():
    s = pd.Series(["sanity_error", "unsupported_jump", "uncategorized"])
    preferred = ["unsupported_jump", "sanity_error", "abs_deviation", "uncategorized"]
    result = _ordered_present_keys(s, preferred)
    assert result == ["unsupported_jump", "sanity_error", "uncategorized"]


def test_ordered_present_keys_extras_appended_sorted():
    s = pd.Series(["z_extra", "sanity_error", "a_extra"])
    preferred = ["sanity_error"]
    result = _ordered_present_keys(s, preferred)
    assert result == ["sanity_error", "a_extra", "z_extra"]


def test_ordered_present_keys_blank_and_nan_excluded():
    s = pd.Series(["sanity_error", None, "", "  "])
    result = _ordered_present_keys(s, ["sanity_error"])
    assert result == ["sanity_error"]


# ---------------------------------------------------------------------------
# _prepare_a3_filter_frame
# ---------------------------------------------------------------------------

def test_prepare_new_schema():
    """신규 스키마: outlier_reason / structure_type 채워진 DataFrame."""
    df = pd.DataFrame({
        "outlier_reason": ["unsupported_jump", "abs_deviation", "sanity_error"],
        "structure_type": ["normal", "leader_or_isolated", "normal"],
    })
    result, has_reason, has_structure = _prepare_a3_filter_frame(df)

    assert has_reason
    assert has_structure
    assert list(result["_reason_filter_key"]) == ["unsupported_jump", "abs_deviation", "sanity_error"]
    assert list(result["판정사유"]) == ["지지 없는 단발 점프", "절대 금액 이탈 (±3억)", "입력/단위 오류"]
    assert list(result["_structure_filter_key"]) == ["normal", "leader_or_isolated", "normal"]
    assert list(result["단지유형"]) == ["일반 단지", "대장/나홀로 단지", "일반 단지"]


def test_prepare_legacy_schema():
    """legacy 스키마: reference_type 만 존재 (outlier_reason 없음)."""
    df = pd.DataFrame({
        "reference_type": ["moving_average_band", "trend_month_robust_band", "sanity_error"],
    })
    result, has_reason, has_structure = _prepare_a3_filter_frame(df)

    assert has_reason is False
    assert has_structure is False
    assert list(result["_reason_filter_key"]) == [
        "legacy_band_outlier",
        "trend_month_robust_band",
        "sanity_error",
    ]
    # 단지유형은 legacy_unknown 폴백
    assert (result["_structure_filter_key"] == "legacy_unknown").all()
    assert (result["단지유형"] == "미분류 (legacy 데이터)").all()


def test_prepare_fallback_schema():
    """컬럼 모두 없는 폴백 DataFrame."""
    df = pd.DataFrame({"price": [100, 200, 300]})
    result, has_reason, has_structure = _prepare_a3_filter_frame(df)

    assert has_reason is False
    assert has_structure is False
    assert (result["_reason_filter_key"] == "uncategorized").all()
    assert (result["판정사유"] == "미분류").all()
    assert (result["_structure_filter_key"] == "legacy_unknown").all()


# ---------------------------------------------------------------------------
# Entry re-export smoke
# ---------------------------------------------------------------------------

def test_entry_reexport_is_same_object():
    """엔트리포인트 re-export 가 a3_filters 의 동일 객체인지 확인한다."""
    from dashboard.pages.page_00_market_snapshot_diagnostics import (
        _resolve_color_col as entry_fn,
    )
    from dashboard.pages.snapshot.a3_filters import _resolve_color_col as direct_fn

    assert entry_fn is direct_fn
