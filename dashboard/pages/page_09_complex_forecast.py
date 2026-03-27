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
    st.header("🏘️ 매매가 예측")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 예측 시차 선택 → ② 정확도 지표 확인 → ③ 예측 그래프 → ④ 중요 변수 확인

        **예측 시차:** 1개월 = 한 달 뒤 가격 예측. 3개월 = 3달 뒤 가격 예측.

        **정확도 지표 해석:**
        - **MAE (만원/㎡)**: 평균 절대 오차. 예측이 실제와 평균 얼마나 다른지. 낮을수록 좋음.
        - **RMSE (만원/㎡)**: 큰 오차를 더 크게 처벌하는 지표. MAE보다 크면 가끔 크게 틀린다는 의미.
        - **MAPE (%)**: 오차율 %. 5% 미만: 매우 정확, 10~20%: 방향 참고 수준.
        - **방향 적중률 (%)**: 오를지 내릴지만 맞힌 비율. 50% = 동전 던지기. 60% 이상이면 의미 있음.

        **중요 변수 막대**: 예측에 가장 큰 영향을 준 변수. 가장 긴 막대 = 이 예측의 핵심 신호.
        """)

    horizon = int(st.radio("예측 시차", options=[1, 3], horizontal=True, key="complex_sale_forecast_horizon"))
    pred_df, metrics, importance_df = _get_sale_forecast(horizon)
    if pred_df.empty:
        st.warning("매매가 예측을 위한 타깃 데이터가 부족합니다.")
        return
    _render_forecast_metrics(metrics)
    st.plotly_chart(build_forecast_chart(pred_df, f"{horizon}개월 선행 매매가 예측", "m2당 매매가"), width="stretch")
    st.plotly_chart(build_importance_chart(importance_df, "매매가 예측 중요 변수"), width="stretch")


def render_complex_rent_forecast() -> None:
    st.header("🏘️ 전세·월세 예측")
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
    st.header("🏘️ 12개월 수익률 예측")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **수익률 예측이란?** 지금 이 가격에 매수하면 12개월 후 가격이 몇 % 변할지 예측.

        **수치 해석:**
        - **수익률 +5%**: 현재 가격 대비 1년 후 약 5% 상승 예상. 5억이면 5.25억 예상.
        - **수익률 -3%**: 현재 가격 대비 1년 후 약 3% 하락 예상.
        - **수익률 0%**: 횡보 예상.

        **⚠️ 중요 주의사항:**
        - 이 수익률은 가격 변화만 반영. 취득세·양도세·중개 수수료는 별도.
        - 매매는 시세 차익, 전세는 보증금 상승분, 월세는 임대료 변동으로 각각 계산.
        - 예측값은 과거 패턴 기반. 미래를 보장하지 않습니다.

        **방향 적중률 60% 이상이면**: 오를지 내릴지 방향 판단에 참고할 만한 수준.
        """)

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
    st.header("🏘️ 전세가율·전환율 예측")
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
    st.header("🏘️ 경기 변화 시나리오 시뮬레이션")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **시나리오 시뮬레이션이란?** 금리·통화량·공급 조건을 내가 직접 바꿔보고, 그 경우 가격이 어떻게 변할지 예측하는 도구.

        **슬라이더 설명:**
        - **기준금리 변화**: +0.25 = 금리 0.25%포인트 인상. -0.5 = 금리 0.5%포인트 인하.
        - **M2 YoY 충격**: +5 = 시중 통화량이 예상보다 5% 더 늘어나는 경우.
        - **입주물량 압력**: +1 = 공급이 평균보다 많아지는 상황 (실제 공급 데이터 대신 추정치 사용).

        **결과 해석:**
        - **막대 오른쪽(양수)**: 해당 시나리오에서 이 단지/지역 가격이 오를 것으로 예측.
        - **막대 왼쪽(음수)**: 해당 시나리오에서 가격이 내릴 것으로 예측.
        - **막대 길이**: 가격 변화 폭. 길수록 영향이 큼.

        **⚠️ 주의:** 입주물량은 현재 실제 공급 데이터 대신 단지 밀도·세대수 기반 추정치를 사용합니다.
        """)

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
