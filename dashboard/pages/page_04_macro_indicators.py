"""Page 04 - Level 3 거시지표 연계 분석."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import get_scope_codes
from analysis.level3 import (
    build_correlation_heatmap,
    build_dual_correlation_heatmaps,
    build_fx_event_chart,
    build_m2_price_chart,
    build_macro_scatter,
    build_rate_lag_chart,
    build_real_price_chart,
    build_scope_frame,
    prepare_combined_correlation,
    prepare_fx_event_study,
    prepare_real_price_index,
)
from dashboard.data_loader import (
    get_scope_option_list,
    load_macro_monthly,
    load_rent_summary,
    load_trade_summary,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_scope_frame(scope_name: str):
    trade_df = load_trade_summary()
    rent_df = load_rent_summary()
    macro_df = load_macro_monthly()
    return build_scope_frame(trade_df, rent_df, macro_df, get_scope_codes(scope_name), scope_name)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_fx_event():
    trade_df = load_trade_summary()
    rent_df = load_rent_summary()
    macro_df = load_macro_monthly()
    combined = build_scope_frame(trade_df, rent_df, macro_df, get_scope_codes("수도권 전체"), "수도권 전체")
    return prepare_fx_event_study(combined)


def render_rate_lag() -> None:
    st.header("Level 3 - 금리 시차 상관")
    scope_name = st.selectbox("분석 범위", get_scope_option_list(), index=0, key="level3_rate_scope")
    combined = _get_scope_frame(scope_name)
    st.plotly_chart(build_rate_lag_chart(combined, scope_name), width="stretch")


def render_m2() -> None:
    st.header("Level 3 - M2 연계")
    scope_name = st.selectbox("분석 범위", get_scope_option_list(), index=0, key="level3_m2_scope")
    combined = _get_scope_frame(scope_name)
    st.plotly_chart(build_m2_price_chart(combined, scope_name), width="stretch")


def render_fx_event() -> None:
    st.header("Level 3 - 환율 이벤트")
    fx_df = _get_fx_event()
    st.plotly_chart(build_fx_event_chart(fx_df), width="stretch")


def render_real_price() -> None:
    st.header("Level 3 - 실질 가격")
    scope_name = st.selectbox("분석 범위", get_scope_option_list(), index=0, key="level3_real_scope")
    combined = _get_scope_frame(scope_name)
    real_df = prepare_real_price_index(combined)
    st.plotly_chart(build_real_price_chart(real_df, scope_name), width="stretch")


def render_correlation() -> None:
    st.header("Level 3 - 복합 상관")
    scope_options = get_scope_option_list()
    scope_name = st.selectbox("기준 범위", scope_options, index=0, key="level3_corr_scope")
    combined = prepare_combined_correlation(_get_scope_frame(scope_name))
    if combined.empty:
        st.info("복합 상관관계 데이터가 없습니다.")
        return

    st.plotly_chart(build_correlation_heatmap(combined), width="stretch")

    compare_col1, compare_col2 = st.columns(2)
    scope_a = compare_col1.selectbox("비교 범위 A", scope_options, index=0, key="level3_corr_scope_a")
    scope_b = compare_col2.selectbox("비교 범위 B", scope_options, index=min(1, len(scope_options) - 1), key="level3_corr_scope_b")
    combined_a = prepare_combined_correlation(_get_scope_frame(scope_a))
    combined_b = prepare_combined_correlation(_get_scope_frame(scope_b))
    if not combined_a.empty and not combined_b.empty:
        st.plotly_chart(build_dual_correlation_heatmaps(combined_a, scope_a, combined_b, scope_b), width="stretch")

    numeric_cols = [column for column in combined.columns if column != "date"]
    x_col = st.selectbox("산점도 X축", numeric_cols, index=0, key="level3_scatter_x")
    y_candidates = [column for column in numeric_cols if column != x_col]
    y_col = st.selectbox("산점도 Y축", y_candidates, index=0, key="level3_scatter_y")
    fig_scatter, reg = build_macro_scatter(combined, x_col, y_col)
    st.plotly_chart(fig_scatter, width="stretch")
    st.metric("R²", f"{reg['r_squared']:.4f}" if reg["r_squared"] == reg["r_squared"] else "N/A")