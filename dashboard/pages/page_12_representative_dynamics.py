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
    st.header("🏡 시간에 따라 변하는 평형 가격 영향력")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이 분석은 무엇인가?** 84형과 59형의 가격 차이(gap)에 영향을 주는 요인들이 시간에 따라 어떻게 바뀌었는지 추적합니다.

        **그래프 해석:**
        - **각 선**: 특정 요인(금리, 통화량 등)이 84/59 가격 격차에 미치는 영향의 시계열 변화.
        - **선이 위로**: 그 시기에 해당 요인이 84형과 59형 격차를 벌리는 방향으로 작용.
        - **선이 아래로**: 그 시기에 해당 요인이 두 평형 격차를 줄이는 방향으로 작용.
        - **선이 0을 교차하는 시점**: 그 요인의 영향 방향이 바뀌는 전환점.

        **💡 팁:** 금리가 급격히 오른 시기에 격차 관련 계수가 어떻게 변했는지 주목해보세요.
        """)

    rolling_df = _get_gap_rolling()
    if rolling_df.empty:
        st.warning("롤링 계수를 계산할 데이터가 없습니다.")
        return
    st.plotly_chart(build_gap_rolling_coefficient_chart(rolling_df), width="stretch")
    st.dataframe(rolling_df, width="stretch", hide_index=True)


def render_representative_panel_fe() -> None:
    st.header("🏡 단지 특성 제거 후 84형·59형 격차 변화")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이 분석이 하는 일:** 단지마다 다른 고유 특성(입지 등)을 제거한 뒤, 순수하게 경기 변화(금리, 통화량)가 84형·59형 가격 격차에 어떤 영향을 주는지 봅니다.

        **막대그래프 해석 (X축: gap 변화, 단위: pp):**
        - **양수 막대**: 해당 경기 지표가 오를 때 84형·59형 격차도 커지는 경향.
          - 예: 금리 상승 시 +2pp → 금리 오를수록 84형이 59형 대비 더 빨리 오름
        - **음수 막대**: 해당 경기 지표가 오를 때 격차가 줄어드는 경향.
          - 예: 통화량 증가 시 -1pp → 돈 풀릴수록 두 평형 가격이 비슷해지는 경향

        **pp (퍼센트포인트):** 비율 간 차이 단위. 격차율이 15%→17%로 올랐으면 2pp 상승.

        **Within R²:** 단지 특성 제거 후 남은 격차 변동의 몇 %를 이 모형이 설명하는지.
        """)

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
    st.header("🏡 경기 국면별 평형 격차 반응")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이 분석이 보여주는 것:** 금리 상승기, 유동성 확장기 등 경기 국면에 따라 84형·59형 가격 격차가 어떻게 달라지는지.

        **막대그래프 해석:**
        - **각 막대**: 해당 경기 국면에서 격차가 평균 대비 얼마나 달라졌는지 (pp 단위).
          - 예: 금리 상승기 +3pp → 금리 오를 때는 84형이 59형 대비 평균보다 3pp 더 올라감
        - **플러스 막대**: 해당 국면에서 대형(84형) 선호가 강해지는 경향.
        - **마이너스 막대**: 해당 국면에서 소형(59형) 선호가 강해지거나 대형이 상대적 약세.

        **💡 투자 팁:** 특정 경기 국면이 예상될 때 어느 평형이 유리한지 미리 파악하는 데 활용할 수 있습니다.
        """)

    regime_df = _get_regime_response()
    if regime_df.empty:
        st.warning("거시 국면 반응을 계산할 데이터가 없습니다.")
        return
    st.plotly_chart(build_regime_response_chart(regime_df), width="stretch")
    st.dataframe(regime_df, width="stretch", hide_index=True)


def render_representative_spillover() -> None:
    st.header("🏡 지역 간 가격 격차 전파")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이 분석이 보여주는 것:** 한 지역의 84형·59형 가격 격차 변화가 다른 지역으로 얼마나 빠르게 퍼지는지.

        **주요 수치 해석:**
        - **선도→추종 lag1 상관 (0~1)**: 선도 지역의 격차 변화가 1개월 후 추종 지역에서 얼마나 비슷하게 나타나는지.
          - 0.6 이상: 강한 전파 효과 (선도 지역 따라가는 패턴 뚜렷)
          - 0.2 이하: 지역별 독립적 움직임
        - **평균 격차 차이 (pp)**: 선도 지역과 추종 지역의 가격 격차 차이. 클수록 두 지역의 시장 특성이 다름.

        **💡 팁:** 선도 지역의 격차 변화를 먼저 파악하면 추종 지역의 향후 움직임을 미리 예상할 수 있습니다.
        """)

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
    st.header("🏡 가격 격차가 평균으로 돌아오는 경향")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **평균회귀(Mean Reversion)란?** 가격 격차가 역사적 평균에서 벗어났을 때 결국 다시 평균으로 돌아오려는 경향.

        **왜 중요할까?** 84형·59형 격차가 역사적으로 평균 20%라면:
        - 격차가 30%로 벌어지면 → 다시 20%로 줄어들 가능성
        - 격차가 10%로 좁혀지면 → 다시 20%로 벌어질 가능성

        **그래프 해석:**
        - **현재 격차 위치**: 역사적 평균 대비 현재 격차가 얼마나 높거나 낮은지.
        - **회귀 속도**: 선이 빠르게 평균 방향으로 꺾이면 격차가 빠르게 정상화되는 시장.
        - **회귀 속도가 느리면**: 격차 확대/축소 상태가 오래 지속되는 경향.

        **⚠️ 주의:** 평균회귀는 경향이지 법칙이 아닙니다. 구조적 변화(재건축 급증, 소형 공급 감소 등)가 있으면 새로운 평균이 형성될 수 있습니다.
        """)

    reversion_df = _get_mean_reversion()
    if reversion_df.empty:
        st.warning("평균회귀 분석 데이터가 없습니다.")
        return
    st.plotly_chart(build_mean_reversion_chart(reversion_df), width="stretch")
    st.dataframe(reversion_df, width="stretch", hide_index=True)
