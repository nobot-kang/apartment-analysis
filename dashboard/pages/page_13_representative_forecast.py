"""Dashboard page for representative 59/84 forecasts and scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.representative_complex_analysis import (
    build_forecast_chart,
    build_importance_chart,
    build_scenario_chart,
    build_scenario_frame,
    build_screening_chart,
    build_screening_frame,
    run_gap_forecast,
    run_sale_band_forecast,
)
from dashboard.data_loader import load_representative_forecast_targets


def _render_forecast_metrics(metrics: dict[str, float]) -> None:
    cols = st.columns(4)
    cols[0].metric("MAE", f"{metrics['mae']:.3f}" if metrics["mae"] == metrics["mae"] else "N/A")
    cols[1].metric("RMSE", f"{metrics['rmse']:.3f}" if metrics["rmse"] == metrics["rmse"] else "N/A")
    cols[2].metric("MAPE", f"{metrics['mape']:.2f}%" if metrics["mape"] == metrics["mape"] else "N/A")
    cols[3].metric("방향 적중률", f"{metrics['directional_accuracy']:.1f}%" if metrics["directional_accuracy"] == metrics["directional_accuracy"] else "N/A")


@st.cache_data(ttl=3600, show_spinner=False)
def _get_sale_forecast(band: int, horizon: int):
    return run_sale_band_forecast(load_representative_forecast_targets(), band, horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_gap_forecast(horizon: int):
    return run_gap_forecast(load_representative_forecast_targets(), horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_screening(horizon: int):
    return build_screening_frame(load_representative_forecast_targets(), horizon=horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_scenario(rate_delta: float, liquidity_delta: float, fx_delta: float):
    return build_scenario_frame(
        load_representative_forecast_targets(),
        rate_delta=rate_delta,
        liquidity_delta=liquidity_delta,
        fx_delta=fx_delta,
    )


def _render_sale_forecast(*, band: int, key_prefix: str) -> None:
    st.header(f"Representative Level 4 - {band}형 평당가 예측")
    horizon = int(st.radio("예측 시차", options=[1, 3, 12], horizontal=True, key=f"{key_prefix}_horizon"))
    pred_df, metrics, importance_df = _get_sale_forecast(band, horizon)
    if pred_df.empty:
        st.warning("예측에 필요한 데이터가 부족합니다.")
        return
    _render_forecast_metrics(metrics)
    st.plotly_chart(build_forecast_chart(pred_df, f"{band}형 {horizon}개월 선행 수익률 예측", "수익률(%)"), width="stretch")
    st.plotly_chart(build_importance_chart(importance_df, f"{band}형 예측 중요 변수"), width="stretch")


def render_representative_sale59_forecast() -> None:
    _render_sale_forecast(band=59, key_prefix="rep_sale59_forecast")


def render_representative_sale84_forecast() -> None:
    _render_sale_forecast(band=84, key_prefix="rep_sale84_forecast")


def render_representative_gap_forecast() -> None:
    st.header("Representative Level 4 - pair gap 비율 예측")
    horizon = int(st.radio("예측 시차", options=[1, 3, 12], horizontal=True, key="rep_gap_forecast_horizon"))
    pred_df, metrics, importance_df = _get_gap_forecast(horizon)
    if pred_df.empty:
        st.warning("pair gap 예측 데이터가 부족합니다.")
        return
    _render_forecast_metrics(metrics)
    st.plotly_chart(build_forecast_chart(pred_df, f"{horizon}개월 선행 84/59 gap 변화 예측", "gap 변화(pp)"), width="stretch")
    st.plotly_chart(build_importance_chart(importance_df, "pair gap 예측 중요 변수"), width="stretch")


def render_representative_screening() -> None:
    st.header("Representative Level 4 - 지역/단지 스크리닝")
    horizon = int(st.radio("기준 시차", options=[1, 3, 12], horizontal=True, key="rep_screening_horizon"))
    region_rank_df, complex_rank_df = _get_screening(horizon)
    if region_rank_df.empty and complex_rank_df.empty:
        st.warning("스크리닝 데이터가 없습니다.")
        return
    if not region_rank_df.empty:
        st.plotly_chart(build_screening_chart(region_rank_df), width="stretch")
    cols = st.columns(2)
    if not region_rank_df.empty:
        cols[0].dataframe(region_rank_df.head(20), width="stretch", hide_index=True)
    if not complex_rank_df.empty:
        cols[1].dataframe(complex_rank_df.head(20), width="stretch", hide_index=True)


def render_representative_scenario() -> None:
    st.header("Representative Level 4 - 대표 평형 spread 시나리오")
    rate_delta = float(st.slider("금리 3개월 변화 충격", min_value=-1.5, max_value=1.5, value=0.25, step=0.25, key="rep_scenario_rate"))
    liquidity_delta = float(st.slider("M2 YoY 충격", min_value=-10.0, max_value=10.0, value=2.0, step=0.5, key="rep_scenario_liquidity"))
    fx_delta = float(st.slider("환율 3개월 변화 충격", min_value=-100.0, max_value=100.0, value=10.0, step=5.0, key="rep_scenario_fx"))
    scenario_df = _get_scenario(rate_delta, liquidity_delta, fx_delta)
    if scenario_df.empty:
        st.warning("시나리오 분석 데이터가 부족합니다.")
        return
    st.caption("기준 archetype과 대단지, 주차 우수, 저밀도, 재건축 잠재력 단지의 gap 반응을 비교합니다.")
    st.plotly_chart(build_scenario_chart(scenario_df), width="stretch")
    st.dataframe(scenario_df, width="stretch", hide_index=True)
