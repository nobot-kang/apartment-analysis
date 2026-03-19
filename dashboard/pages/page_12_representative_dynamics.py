"""Dashboard page for representative 59/84 dynamics and causal lenses."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.representative_complex_analysis import (
    build_effect_chart,
    build_gap_rolling_coefficient_chart,
    build_gap_rolling_coefficient_frame,
    build_mean_reversion_chart,
    build_mean_reversion_frame,
    build_regime_response_chart,
    build_regime_response_frame,
    build_spillover_chart,
    build_spillover_frame,
    run_gap_panel_fixed_effects,
)
from dashboard.data_loader import load_representative_forecast_targets, load_representative_region_monthly


@st.cache_data(ttl=3600, show_spinner=False)
def _get_gap_rolling() -> pd.DataFrame:
    return build_gap_rolling_coefficient_frame(load_representative_forecast_targets())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_gap_panel_fe():
    return run_gap_panel_fixed_effects(load_representative_forecast_targets())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_regime_response() -> pd.DataFrame:
    return build_regime_response_frame(load_representative_forecast_targets())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_spillover():
    return build_spillover_frame(load_representative_region_monthly())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_mean_reversion() -> pd.DataFrame:
    return build_mean_reversion_frame(load_representative_forecast_targets())


def render_representative_rolling_coefficients() -> None:
    st.header("Representative Level 3 - pair gap 롤링 계수")
    rolling_df = _get_gap_rolling()
    if rolling_df.empty:
        st.warning("롤링 계수를 계산할 데이터가 없습니다.")
        return
    st.plotly_chart(build_gap_rolling_coefficient_chart(rolling_df), width="stretch")
    st.dataframe(rolling_df, width="stretch", hide_index=True)


def render_representative_panel_fe() -> None:
    st.header("Representative Level 3 - pair gap 패널 고정효과")
    result = _get_gap_panel_fe()
    if result.coefficients.empty:
        st.warning("패널 고정효과 모형을 추정할 데이터가 부족합니다.")
        return
    cols = st.columns(2)
    cols[0].metric("관측치 수", f"{result.metrics['n_obs']:,.0f}")
    cols[1].metric("Within R²", f"{result.metrics['r_squared']:.3f}" if result.metrics["r_squared"] == result.metrics["r_squared"] else "N/A")
    st.plotly_chart(
        build_effect_chart(
            result.coefficients,
            "동일 단지 내부 84/59 gap 변화의 거시 반응",
            effect_column="effect_value",
            xaxis_title="gap 변화(pp)",
        ),
        width="stretch",
    )
    st.dataframe(result.coefficients, width="stretch", hide_index=True)


def render_representative_regime_response() -> None:
    st.header("Representative Level 3 - 거시 국면별 spread 반응")
    regime_df = _get_regime_response()
    if regime_df.empty:
        st.warning("거시 국면 반응을 계산할 데이터가 없습니다.")
        return
    st.plotly_chart(build_regime_response_chart(regime_df), width="stretch")
    st.dataframe(regime_df, width="stretch", hide_index=True)


def render_representative_spillover() -> None:
    st.header("Representative Level 3 - 지역 spread 확산")
    spillover_df, metrics = _get_spillover()
    if spillover_df.empty:
        st.warning("확산 분석 데이터가 없습니다.")
        return
    cols = st.columns(2)
    cols[0].metric("선도→추종 lag1 상관", f"{metrics['lag1_corr']:.3f}" if metrics["lag1_corr"] == metrics["lag1_corr"] else "N/A")
    cols[1].metric("평균 spread 차이", f"{metrics['spread_pp']:.2f}pp" if metrics["spread_pp"] == metrics["spread_pp"] else "N/A")
    st.plotly_chart(build_spillover_chart(spillover_df), width="stretch")
    st.dataframe(spillover_df.tail(36).sort_values("date", ascending=False), width="stretch", hide_index=True)


def render_representative_mean_reversion() -> None:
    st.header("Representative Level 3 - pair gap 평균회귀")
    reversion_df = _get_mean_reversion()
    if reversion_df.empty:
        st.warning("평균회귀 분석 데이터가 없습니다.")
        return
    st.plotly_chart(build_mean_reversion_chart(reversion_df), width="stretch")
    st.dataframe(reversion_df, width="stretch", hide_index=True)
