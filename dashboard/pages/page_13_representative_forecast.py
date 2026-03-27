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
    st.header(f"🏡 {band}형 평당가 수익률 예측")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown(f"""
        **보는 순서:** ① 예측 시차 선택 → ② 정확도 지표 확인 → ③ 예측 그래프 → ④ 중요 변수 확인

        **예측 시차:** 1개월 = 한 달 뒤 수익률 예측. 3개월 = 3달 뒤. 12개월 = 1년 뒤.

        **수익률 그래프 해석:**
        - **양수 (%)**: 향후 해당 기간 동안 가격 상승 예측. 예: +5% → 1년 후 지금보다 5% 오를 것으로 예측.
        - **음수 (%)**: 가격 하락 예측.
        - **파란 선(실제) vs 빨간 선(예측)**: 두 선이 가까울수록 모델이 잘 맞은 것.

        **정확도 지표 해석:**
        - **MAE / RMSE**: 예측 오차 크기. 낮을수록 좋음.
        - **MAPE (%)**: 오차율. 5% 미만: 매우 정확. 10~20%: 방향성만 참고.
        - **방향 적중률 (%)**: 오를지 내릴지만 맞힌 비율. 60% 이상이면 의미 있는 예측.

        **중요 변수:** 가장 긴 막대 = 이 {band}형 예측에서 가장 큰 신호를 주는 변수.

        **⚠️ 주의:** 예측은 과거 패턴 기반입니다. 수수료·세금은 별도이며, 투자 판단의 보조 도구로만 활용하세요.
        """)

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
    st.header("🏡 84형·59형 가격 차이 예측")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이 예측이 보여주는 것:** 향후 몇 달 동안 84형·59형의 평당가 격차(gap)가 몇 %포인트(pp) 변할지 예측.

        **수치 해석:**
        - **gap 변화 +2pp**: 현재 격차보다 2%포인트 더 벌어질 것으로 예측. 현재 격차 15%이면 → 17%로 확대 예상.
        - **gap 변화 -1pp**: 격차가 좁혀질 것으로 예측. 15% → 14%로 축소.
        - **gap 변화 0**: 격차 유지 예측.

        **이걸 어디에 쓰나?**
        - gap이 벌어질 것으로 예측되면 → 84형이 59형보다 상대적으로 더 오를 전망
        - gap이 좁혀질 것으로 예측되면 → 59형이 84형보다 상대적으로 강세

        **방향 적중률이 중요:** gap 예측은 방향(벌어질지/좁혀질지)이 실제 투자 판단에 더 중요합니다.
        """)

    horizon = int(st.radio("예측 시차", options=[1, 3, 12], horizontal=True, key="rep_gap_forecast_horizon"))
    pred_df, metrics, importance_df = _get_gap_forecast(horizon)
    if pred_df.empty:
        st.warning("pair gap 예측 데이터가 부족합니다.")
        return
    _render_forecast_metrics(metrics)
    st.plotly_chart(build_forecast_chart(pred_df, f"{horizon}개월 선행 84/59 gap 변화 예측", "gap 변화(pp)"), width="stretch")
    st.plotly_chart(build_importance_chart(importance_df, "pair gap 예측 중요 변수"), width="stretch")


def render_representative_screening() -> None:
    st.header("🏡 관심 지역·단지 순위 스크리닝")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **스크리닝이란?** 예측 수익률 기준으로 지역과 단지를 순위 매겨 보여주는 화면.

        **보는 순서:** ① 기준 시차 선택 → ② 지역 순위 확인 → ③ 단지별 순위 확인

        **그래프/표 해석:**
        - **상위 순위**: 해당 기간 동안 예측 수익률이 높은 지역·단지.
        - **막대 길이**: 길수록 예측 수익률이 높음.

        **⚠️ 반드시 주의:**
        - 이 순위는 AI 모델의 예측값 기반입니다. 보장이 아닙니다.
        - 상위 지역이라도 개인 재정 상황, 세금, 시장 리스크를 반드시 고려하세요.
        - 예측 정확도(MAPE, 방향 적중률)가 낮은 경우 순위를 신뢰하기 어렵습니다.

        **💡 활용법:** '내가 관심 있는 지역이 상위 20위 안에 드는지' 확인하는 용도로 활용하세요.
        """)

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
    st.header("🏡 경기 변화 시나리오 시뮬레이션")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **시나리오 시뮬레이션이란?** 금리·통화량·환율을 내가 직접 조정해보고, 그때 84형·59형 가격 격차가 어떻게 바뀔지 예측하는 도구.

        **슬라이더 설명:**
        - **금리 변화**: +0.5 = 기준금리 0.5%포인트 인상 충격.
        - **M2 YoY 충격**: +5 = 시중 통화량이 예상보다 5% 더 늘어나는 경우.
        - **환율 변화 (원/달러)**: +50 = 달러 대비 원화가 50원 약세. -50 = 원화 50원 강세.

        **결과 해석:**
        - 각 단지 유형(기본·대단지·주차우수·저밀도·재건축 잠재력)별로 격차 반응을 비교.
        - 막대 오른쪽 = 84형이 59형보다 더 오르는 방향. 왼쪽 = 59형이 상대적 강세.

        **💡 팁:** 금리를 +1%p로 올렸을 때 재건축 잠재 단지의 반응이 어떻게 다른지 비교해보세요.
        """)

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
