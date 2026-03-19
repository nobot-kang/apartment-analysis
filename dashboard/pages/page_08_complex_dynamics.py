"""Dashboard page for complex-level dynamics and causal lenses."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.complex_analysis import (
    build_effect_chart,
    build_latest_snapshot,
    build_redevelopment_chart,
    build_redevelopment_frame,
    build_regime_premium_chart,
    build_regime_premium_frame,
    build_rolling_coefficient_chart,
    build_rolling_coefficient_frame,
    build_spillover_chart,
    build_spillover_frame,
    build_yearly_snapshot,
    run_panel_fixed_effects,
)
from dashboard.data_loader import load_complex_forecast_targets, load_complex_monthly_panel


@st.cache_data(ttl=3600, show_spinner=False)
def _get_yearly_snapshot():
    return build_yearly_snapshot(load_complex_monthly_panel())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_rolling_coefficients():
    return build_rolling_coefficient_frame(_get_yearly_snapshot())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_panel_fe():
    return run_panel_fixed_effects(load_complex_monthly_panel())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_regime_frame():
    return build_regime_premium_frame(load_complex_monthly_panel())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_redevelopment_frame(months: int):
    return build_redevelopment_frame(
        build_latest_snapshot(load_complex_monthly_panel(), months=months),
        load_complex_forecast_targets(),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_spillover():
    return build_spillover_frame(load_complex_monthly_panel())


def render_complex_rolling_coefficients() -> None:
    st.header("Complex Level 3 - 롤링 계수 분석")
    rolling_df = _get_rolling_coefficients()
    if rolling_df.empty:
        st.warning("연도별 프리미엄 변화를 계산할 데이터가 없습니다.")
        return
    st.plotly_chart(build_rolling_coefficient_chart(rolling_df), width="stretch")
    st.dataframe(rolling_df, width="stretch", hide_index=True)


def render_complex_panel_fe() -> None:
    st.header("Complex Level 3 - 단지 패널 고정효과 모형")
    result = _get_panel_fe()
    if result.coefficients.empty:
        st.warning("패널 고정효과 모형을 추정할 데이터가 부족합니다.")
        return
    cols = st.columns(2)
    cols[0].metric("관측치 수", f"{result.metrics['n_obs']:,.0f}")
    cols[1].metric("Within R²", f"{result.metrics['r_squared']:.3f}" if result.metrics["r_squared"] == result.metrics["r_squared"] else "N/A")
    st.plotly_chart(
        build_effect_chart(
            result.coefficients,
            "동일 단지 내부 가격 변화에 대한 거시 상호작용",
            effect_column="effect_value",
            xaxis_title="추가 YoY(pp)",
        ),
        width="stretch",
    )
    st.dataframe(result.coefficients, width="stretch", hide_index=True)


def render_complex_macro_interactions() -> None:
    st.header("Complex Level 3 - 거시 상호작용 분석")
    regime_df = _get_regime_frame()
    if regime_df.empty:
        st.warning("거시 상호작용 분석 데이터가 없습니다.")
        return
    st.plotly_chart(build_regime_premium_chart(regime_df), width="stretch")
    st.dataframe(regime_df, width="stretch", hide_index=True)


def render_complex_redevelopment() -> None:
    st.header("Complex Level 3 - 재건축 옵션 분석")
    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_redevelopment_months"))
    redevelopment_df = _get_redevelopment_frame(months)
    if redevelopment_df.empty:
        st.warning("재건축 옵션 분석 데이터가 없습니다.")
        return
    st.plotly_chart(build_redevelopment_chart(redevelopment_df), width="stretch")
    st.dataframe(redevelopment_df, width="stretch", hide_index=True)


def render_complex_spillover() -> None:
    st.header("Complex Level 3 - 확산/스필오버 분석")
    spillover_df, metrics = _get_spillover()
    if spillover_df.empty:
        st.warning("확산 효과를 계산할 데이터가 없습니다.")
        return
    cols = st.columns(2)
    cols[0].metric("1개월 선행 상관", f"{metrics['lag1_corr']:.3f}" if metrics["lag1_corr"] == metrics["lag1_corr"] else "N/A")
    cols[1].metric("선도-주변 스프레드", f"{metrics['spread_pp']:.2f}pp" if metrics["spread_pp"] == metrics["spread_pp"] else "N/A")
    st.plotly_chart(build_spillover_chart(spillover_df), width="stretch")
