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
    st.header("🏘️ 시간에 따라 변하는 가격 영향력")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **롤링 계수란?** 특정 요인(면적, 층수 등)이 가격에 미치는 영향이 연도별로 어떻게 변했는지 추적하는 분석.

        **그래프 해석:**
        - **각 선**: 특정 단지 특성의 가격 영향력이 시간에 따라 변하는 추이.
        - **선이 위로 올라가는 구간**: 그 특성의 가격 영향력이 커지는 시기.
          - 예: 주차 계수가 2020년 이후 급등 → 주차 중요성이 최근 들어 훨씬 커짐
        - **선이 내려가는 구간**: 그 특성의 가격 영향력이 약해지는 시기.
        - **선이 0을 교차하는 시점**: 영향 방향이 바뀌는 전환점.

        **💡 팁:** 금리 급등·급락, 공급 쇼크 등이 있었던 연도 주변에서 특이한 변화가 자주 나타납니다.
        """)

    rolling_df = _get_rolling_coefficients()
    if rolling_df.empty:
        st.warning("연도별 프리미엄 변화를 계산할 데이터가 없습니다.")
        return
    st.plotly_chart(build_rolling_coefficient_chart(rolling_df), width="stretch")
    st.dataframe(rolling_df, width="stretch", hide_index=True)


def render_complex_panel_fe() -> None:
    st.header("🏘️ 단지 고유 특성 제거 후 순수 시장 효과")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **패널 고정효과 모형이란?** 단지마다 다른 고유한 특성(입지, 브랜드 등 변하지 않는 요소)을 수학적으로 제거하고, 순수하게 시장(금리·경기 등)이 가격 변화에 미치는 영향만 봅니다.

        **비유:** 키가 다른 학생들의 성장속도를 비교할 때, 원래 키 차이는 빼고 '1년 동안 얼마나 자랐는지'만 비교하는 것.

        **막대그래프 해석 (X축: 추가 YoY, 단위: pp):**
        - **양수 막대**: 해당 거시 지표가 오를 때 단지 내 가격도 오르는 경향.
        - **음수 막대**: 해당 거시 지표가 오를 때 단지 내 가격이 내리는 경향.
        - **pp (퍼센트포인트)**: 비율 간의 차이. 예: 가격 상승률이 3%→5%로 올랐으면 2pp 상승.

        **Within R²:** 단지 특성을 제거한 후 남은 가격 변동 중 이 모형이 설명하는 비율. 0.3도 의미 있는 수준.
        """)

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
    st.header("🏘️ 경기 국면별 단지 가격 반응")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **거시 국면이란?** 금리·통화량·거래량 등 경기 지표의 조합에 따라 시장을 '금리 상승기', '유동성 확장기' 등으로 나눈 것.

        **그래프 해석:**
        - **각 막대**: 해당 경기 국면에서 단지 특성의 프리미엄이 평균 대비 얼마나 달라졌는지.
          - 예: 금리 상승기에 주차 프리미엄 -3pp → 금리 오를 때는 주차의 가격 영향이 약해짐
        - **막대가 플러스**: 해당 국면에서 그 특성의 가격 영향이 강해짐.
        - **막대가 마이너스**: 해당 국면에서 그 특성의 가격 영향이 약해짐.

        **💡 팁:** 이 그래프를 보면 '어떤 경기 상황에서 어떤 단지가 유리한지' 파악할 수 있습니다.
        """)

    regime_df = _get_regime_frame()
    if regime_df.empty:
        st.warning("거시 상호작용 분석 데이터가 없습니다.")
        return
    st.plotly_chart(build_regime_premium_chart(regime_df), width="stretch")
    st.dataframe(regime_df, width="stretch", hide_index=True)


def render_complex_redevelopment() -> None:
    st.header("🏘️ 재건축 기대감이 가격에 미치는 영향")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **재건축 옵션 가치란?** 오래된 아파트는 언젠가 재건축될 것이라는 기대감이 현재 가격에 포함되어 있습니다. 이 '기대감 프리미엄'을 수치화한 것.

        **그래프 해석:**
        - **X축 (건축연도)**: 오래된 단지일수록 왼쪽.
        - **Y축 (가격 프리미엄)**: 같은 입지·규모 대비 얼마나 더 비싸게 거래되는지.
        - **오래됐는데 비쌈**: 재건축 기대감이 반영된 것. 건물 가치보다 '땅 + 재건축 기대'로 가격이 결정됨.
        - **연식이 너무 최근이거나 너무 오래된 단지**: 재건축 기대가 낮음(신축은 아직 이름, 너무 오래된 건 규제 지연 우려).

        **💡 팁:** 재건축 연한(보통 30년)에 가까운 단지들의 프리미엄을 특히 주목하세요.
        """)

    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_redevelopment_months"))
    redevelopment_df = _get_redevelopment_frame(months)
    if redevelopment_df.empty:
        st.warning("재건축 옵션 분석 데이터가 없습니다.")
        return
    st.plotly_chart(build_redevelopment_chart(redevelopment_df), width="stretch")
    st.dataframe(redevelopment_df, width="stretch", hide_index=True)


def render_complex_spillover() -> None:
    st.header("🏘️ 인근 단지 가격 상승이 퍼지는 속도")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **확산(Spillover) 효과란?** 특정 단지의 가격이 오르면 인근 단지 가격도 뒤따라 오르는 현상. 마치 물이 퍼지듯 가격 상승이 전파됨.

        **주요 수치 해석:**
        - **1개월 선행 상관**: 선도 단지(먼저 오르는 단지) 가격이 오른 뒤 1개월 후 주변 단지도 따라 오르는 강도.
          - 예: 0.6 → 선도 단지가 오르면 1개월 후 주변 단지도 강하게 따라가는 경향
          - 0.2 이하: 확산 효과 미약
        - **선도-주변 스프레드 (pp)**: 선도 단지와 주변 단지의 가격 변화율 차이.
          - 양수: 선도 단지가 주변보다 먼저, 더 많이 오름
          - 클수록 선도 단지의 '선도 효과'가 강함

        **💡 팁:** 선도 단지를 파악하면 주변 지역 가격 변화를 미리 예측하는 데 도움이 됩니다.
        """)

    spillover_df, metrics = _get_spillover()
    if spillover_df.empty:
        st.warning("확산 효과를 계산할 데이터가 없습니다.")
        return
    cols = st.columns(2)
    cols[0].metric("1개월 선행 상관", f"{metrics['lag1_corr']:.3f}" if metrics["lag1_corr"] == metrics["lag1_corr"] else "N/A")
    cols[1].metric("선도-주변 스프레드", f"{metrics['spread_pp']:.2f}pp" if metrics["spread_pp"] == metrics["spread_pp"] else "N/A")
    st.plotly_chart(build_spillover_chart(spillover_df), width="stretch")
