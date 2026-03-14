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
    load_combined_correlation,
    load_fx_event_study,
    load_real_price_index,
)
from dashboard.data_loader import load_trade_summary


@st.cache_data(ttl=3600)
def _get_combined(scope_name: str):
    return load_combined_correlation(get_scope_codes(scope_name), scope_name)


@st.cache_data(ttl=3600)
def _get_real_price(scope_name: str):
    return load_real_price_index(get_scope_codes(scope_name), scope_name)


@st.cache_data(ttl=3600)
def _get_fx_event():
    return load_fx_event_study()


def render() -> None:
    st.header("Level 3 - 거시지표 연계 분석")

    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    scope_options = ["서울 전체", "경기 전체", "수도권 전체", *sorted(trade_df["_region_name"].unique())]
    scope_name = st.selectbox("분석 범위", scope_options, index=0)
    combined = _get_combined(scope_name)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["금리 시차 상관", "M2 연계", "환율 이벤트", "실질 가격", "복합 상관"])

    with tab1:
        st.plotly_chart(build_rate_lag_chart(get_scope_codes(scope_name), scope_name), width="stretch")

    with tab2:
        st.plotly_chart(build_m2_price_chart(get_scope_codes(scope_name), scope_name), width="stretch")

    with tab3:
        fx_df = _get_fx_event()
        st.plotly_chart(build_fx_event_chart(fx_df), width="stretch")

    with tab4:
        real_df = _get_real_price(scope_name)
        st.plotly_chart(build_real_price_chart(real_df, scope_name), width="stretch")

    with tab5:
        if combined.empty:
            st.info("복합 상관관계 데이터가 없습니다.")
        else:
            st.plotly_chart(build_correlation_heatmap(combined), width="stretch")

            compare_col1, compare_col2 = st.columns(2)
            scope_a = compare_col1.selectbox("비교 범위 A", scope_options, index=0)
            scope_b = compare_col2.selectbox("비교 범위 B", scope_options, index=min(1, len(scope_options) - 1))
            combined_a = _get_combined(scope_a)
            combined_b = _get_combined(scope_b)
            if not combined_a.empty and not combined_b.empty:
                st.plotly_chart(build_dual_correlation_heatmaps(combined_a, scope_a, combined_b, scope_b), width="stretch")

            numeric_cols = [col for col in combined.columns if col != "date"]
            x_col = st.selectbox("산점도 X축", numeric_cols, index=0)
            y_candidates = [col for col in numeric_cols if col != x_col]
            y_col = st.selectbox("산점도 Y축", y_candidates, index=0)
            fig_scatter, reg = build_macro_scatter(combined, x_col, y_col)
            st.plotly_chart(fig_scatter, width="stretch")
            st.metric("R²", f"{reg['r_squared']:.4f}" if reg["r_squared"] == reg["r_squared"] else "N/A")

