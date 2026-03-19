"""Dashboard page for complex-level forecasts and scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.complex_analysis import (
    build_forecast_chart,
    build_importance_chart,
    build_scenario_chart,
    build_scenario_frame,
    run_ratio_forecast,
    run_rent_forecast,
    run_return_forecast,
    run_sale_forecast,
)
from dashboard.data_loader import load_complex_forecast_targets


@st.cache_data(ttl=3600, show_spinner=False)
def _get_sale_forecast(horizon: int):
    return run_sale_forecast(load_complex_forecast_targets(), horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_rent_forecast(metric: str, horizon: int):
    return run_rent_forecast(load_complex_forecast_targets(), metric, horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_return_forecast(metric: str):
    return run_return_forecast(load_complex_forecast_targets(), metric)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_ratio_forecast(metric: str, horizon: int):
    return run_ratio_forecast(load_complex_forecast_targets(), metric, horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_scenario(rate_delta: float, liquidity_delta: float, supply_delta: float):
    return build_scenario_frame(
        load_complex_forecast_targets(),
        rate_delta=rate_delta,
        liquidity_delta=liquidity_delta,
        supply_delta=supply_delta,
    )


def _render_forecast_metrics(metrics: dict[str, float]) -> None:
    cols = st.columns(4)
    cols[0].metric("MAE", f"{metrics['mae']:.3f}" if metrics["mae"] == metrics["mae"] else "N/A")
    cols[1].metric("RMSE", f"{metrics['rmse']:.3f}" if metrics["rmse"] == metrics["rmse"] else "N/A")
    cols[2].metric("MAPE", f"{metrics['mape']:.2f}%" if metrics["mape"] == metrics["mape"] else "N/A")
    cols[3].metric("방향 적중률", f"{metrics['directional_accuracy']:.1f}%" if metrics["directional_accuracy"] == metrics["directional_accuracy"] else "N/A")


def render_complex_sale_forecast() -> None:
    st.header("Complex Level 4 - 단기 매매가 예측")
    horizon = int(st.radio("예측 시차", options=[1, 3], horizontal=True, key="complex_sale_forecast_horizon"))
    pred_df, metrics, importance_df = _get_sale_forecast(horizon)
    if pred_df.empty:
        st.warning("매매가 예측을 위한 타깃 데이터가 부족합니다.")
        return
    _render_forecast_metrics(metrics)
    st.plotly_chart(build_forecast_chart(pred_df, f"{horizon}개월 선행 매매가 예측", "m2당 매매가"), width="stretch")
    st.plotly_chart(build_importance_chart(importance_df, "매매가 예측 중요 변수"), width="stretch")


def render_complex_rent_forecast() -> None:
    st.header("Complex Level 4 - 단기 전세/월세 예측")
    metric = st.radio(
        "예측 대상",
        options=["jeonse", "wolse"],
        horizontal=True,
        format_func=lambda value: "전세" if value == "jeonse" else "월세",
        key="complex_rent_forecast_metric",
    )
    horizon = int(st.radio("예측 시차", options=[1, 3], horizontal=True, key="complex_rent_forecast_horizon"))
    pred_df, metrics, importance_df = _get_rent_forecast(metric, horizon)
    if pred_df.empty:
        st.warning("임대 예측을 위한 타깃 데이터가 부족합니다.")
        return
    _render_forecast_metrics(metrics)
    yaxis = "m2당 전세가" if metric == "jeonse" else "m2당 월세"
    title = f"{horizon}개월 선행 {'전세' if metric == 'jeonse' else '월세'} 예측"
    st.plotly_chart(build_forecast_chart(pred_df, title, yaxis), width="stretch")
    st.plotly_chart(build_importance_chart(importance_df, "임대료 예측 중요 변수"), width="stretch")


def render_complex_return_forecast() -> None:
    st.header("Complex Level 4 - 12개월 수익률 예측")
    metric = st.radio(
        "수익률 대상",
        options=["trade", "jeonse", "wolse"],
        horizontal=True,
        format_func=lambda value: {"trade": "매매", "jeonse": "전세", "wolse": "월세"}[value],
        key="complex_return_forecast_metric",
    )
    pred_df, metrics, importance_df = _get_return_forecast(metric)
    if pred_df.empty:
        st.warning("12개월 수익률 예측을 위한 데이터가 부족합니다.")
        return
    _render_forecast_metrics(metrics)
    st.plotly_chart(build_forecast_chart(pred_df, "향후 12개월 수익률 예측", "수익률(%)"), width="stretch")
    st.plotly_chart(build_importance_chart(importance_df, "수익률 예측 중요 변수"), width="stretch")


def render_complex_ratio_forecast() -> None:
    st.header("Complex Level 4 - 전세가율/전환율 예측")
    metric = st.radio(
        "비율 대상",
        options=["jeonse_ratio", "conversion_rate"],
        horizontal=True,
        format_func=lambda value: "전세가율" if value == "jeonse_ratio" else "전월세 전환율",
        key="complex_ratio_forecast_metric",
    )
    horizon = int(st.radio("예측 시차", options=[1, 3], horizontal=True, key="complex_ratio_forecast_horizon"))
    pred_df, metrics, importance_df = _get_ratio_forecast(metric, horizon)
    if pred_df.empty:
        st.warning("비율 예측을 위한 타깃 데이터가 부족합니다.")
        return
    _render_forecast_metrics(metrics)
    title = f"{horizon}개월 선행 {'전세가율' if metric == 'jeonse_ratio' else '전월세 전환율'} 예측"
    yaxis = "비율(%)"
    st.plotly_chart(build_forecast_chart(pred_df, title, yaxis), width="stretch")
    st.plotly_chart(build_importance_chart(importance_df, "비율 예측 중요 변수"), width="stretch")


def render_complex_scenario() -> None:
    st.header("Complex Level 4 - 시나리오 시뮬레이터")
    rate_delta = float(st.slider("기준금리 3개월 변화 충격", min_value=-1.5, max_value=1.5, value=0.25, step=0.25, key="complex_scenario_rate"))
    liquidity_delta = float(st.slider("M2 YoY 충격", min_value=-10.0, max_value=10.0, value=2.0, step=0.5, key="complex_scenario_liquidity"))
    supply_delta = float(st.slider("입주물량 압력(프록시)", min_value=-5.0, max_value=5.0, value=1.0, step=0.5, key="complex_scenario_supply"))
    scenario_df = _get_scenario(rate_delta, liquidity_delta, supply_delta)
    if scenario_df.empty:
        st.warning("시나리오 분석을 위한 데이터가 부족합니다.")
        return
    st.caption("입주물량 항목은 현재 별도 공급 데이터가 없어 밀도/세대수 노출도를 이용한 프록시로 계산합니다.")
    st.plotly_chart(build_scenario_chart(scenario_df), width="stretch")
    st.dataframe(scenario_df, width="stretch", hide_index=True)
