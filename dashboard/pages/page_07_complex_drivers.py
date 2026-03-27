"""Dashboard page for complex-level pricing drivers."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.complex_analysis import (
    build_effect_chart,
    build_heterogeneity_chart,
    build_heterogeneity_frame,
    build_latest_snapshot,
    build_liquidity_bucket_frame,
    build_liquidity_chart,
    build_yearly_snapshot,
    run_jeonse_hedonic,
    run_liquidity_model,
    run_sale_hedonic,
    run_wolse_hedonic,
)
from dashboard.data_loader import load_complex_monthly_panel


@st.cache_data(ttl=3600, show_spinner=False)
def _get_snapshot(months: int):
    return build_latest_snapshot(load_complex_monthly_panel(), months=months)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_yearly_snapshot():
    return build_yearly_snapshot(load_complex_monthly_panel())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_sale_hedonic(months: int):
    return run_sale_hedonic(_get_snapshot(months))


@st.cache_data(ttl=3600, show_spinner=False)
def _get_jeonse_hedonic(months: int):
    return run_jeonse_hedonic(_get_snapshot(months))


@st.cache_data(ttl=3600, show_spinner=False)
def _get_wolse_hedonic(months: int):
    return run_wolse_hedonic(_get_snapshot(months))


@st.cache_data(ttl=3600, show_spinner=False)
def _get_heterogeneity(months: int):
    return build_heterogeneity_frame(_get_snapshot(months))


@st.cache_data(ttl=3600, show_spinner=False)
def _get_liquidity():
    yearly_df = _get_yearly_snapshot()
    return run_liquidity_model(yearly_df), build_liquidity_bucket_frame(yearly_df)


def _render_model_result(title: str, result) -> None:
    cols = st.columns(2)
    cols[0].metric("관측치 수", f"{result.metrics['n_obs']:,.0f}")
    cols[1].metric("R²", f"{result.metrics['r_squared']:.3f}" if result.metrics["r_squared"] == result.metrics["r_squared"] else "N/A")
    st.plotly_chart(build_effect_chart(result.coefficients, title), width="stretch")
    if not result.coefficients.empty:
        st.dataframe(result.coefficients, width="stretch", hide_index=True)


def render_complex_sale_hedonic() -> None:
    st.header("🏘️ 매매가 구성 요인 분석")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **헤도닉 모형이란?** 아파트 가격을 여러 특성(면적, 층수, 세대수 등)으로 분해하는 통계 분석. 각 특성이 가격에 얼마나 기여하는지 수치로 보여줍니다.

        **막대그래프 해석 (계수값):**
        - **막대가 오른쪽(양수)**: 해당 특성이 클수록 가격이 오름.
          - 예: 면적 계수 +0.5 → 면적이 1% 늘면 가격이 약 0.5% 오름
        - **막대가 왼쪽(음수)**: 해당 특성이 클수록 가격이 내림.
          - 예: 건축연도(구축) 계수 -0.3 → 오래될수록 가격 낮아지는 경향
        - **가장 긴 막대**: 가격에 가장 큰 영향을 주는 요인.
        - **오차 막대(작은 선)**: 불확실성 범위. 오차 막대가 0을 포함하면 통계적으로 불확실.

        **R² 해석:** 이 모형이 단지 간 가격 차이의 몇 %를 설명하는지. 0.7 이상이면 잘 맞는 모형.

        **관측치 수:** 분석에 사용된 단지 수. 많을수록 결과 신뢰도 높음.
        """)

    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_sale_hedonic_months"))
    result = _get_sale_hedonic(months)
    if result.coefficients.empty:
        st.warning("매매 헤도닉 모형을 추정할 데이터가 부족합니다.")
        return
    _render_model_result("매매 가격 구성 요소", result)


def render_complex_jeonse_hedonic() -> None:
    st.header("🏘️ 전세가 구성 요인 분석")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **매매 분석과 같은 방식으로 읽되, 전세 보증금을 설명하는 요인을 보여줍니다.**

        **매매 vs 전세 계수 비교 팁:**
        - 두 분석에서 같은 요인의 계수가 비슷하면 → 매매와 전세가 같은 논리로 가격이 결정됨
        - 특정 요인이 전세에서만 크게 나타나면 → 임차인이 특히 그 특성을 중요시한다는 의미

        **막대그래프 해석:**
        - 오른쪽(양수) 막대 = 해당 특성이 클수록 전세 보증금이 높아짐
        - 왼쪽(음수) 막대 = 해당 특성이 클수록 전세 보증금이 낮아짐
        - 가장 긴 막대 = 전세 가격에 가장 큰 영향을 주는 요인

        **R²:** 이 모형이 단지 간 전세가 차이의 몇 %를 설명하는지.
        """)

    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_jeonse_hedonic_months"))
    result = _get_jeonse_hedonic(months)
    if result.coefficients.empty:
        st.warning("전세 헤도닉 모형을 추정할 데이터가 부족합니다.")
        return
    _render_model_result("전세 가격 구성 요소", result)


def render_complex_wolse_hedonic() -> None:
    st.header("🏘️ 월세 구성 요인 분석")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **월세 보증금이 아닌 월 임대료를 설명하는 요인을 분석합니다.**

        **막대그래프 해석:**
        - 오른쪽(양수) 막대 = 해당 특성이 클수록 월 임대료가 높아짐
        - 왼쪽(음수) 막대 = 해당 특성이 클수록 월 임대료가 낮아짐
        - 가장 긴 막대 = 월세 가격에 가장 큰 영향을 주는 요인

        **매매·전세·월세 3가지 분석을 비교하면:**
        - 같은 요인이 세 시장 모두에서 중요하면 → 그 특성이 '보편적 가격 요인'
        - 월세에서만 특정 요인이 크면 → 임차인(세입자)이 그 특성을 특히 중요하게 여기는 것

        **R²:** 이 모형이 단지 간 월세 차이의 몇 %를 설명하는지.
        """)

    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_wolse_hedonic_months"))
    result = _get_wolse_hedonic(months)
    if result.coefficients.empty:
        st.warning("월세 헤도닉 모형을 추정할 데이터가 부족합니다.")
        return
    _render_model_result("월세 가격 구성 요소", result)


def render_complex_heterogeneity() -> None:
    st.header("🏘️ 가격대별 반응 차이 (싼 단지 vs 비싼 단지)")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이 분석이 필요한 이유:** 같은 요인(예: 면적)이 저가 단지와 고가 단지에서 가격에 미치는 영향이 다를 수 있습니다.

        **분위수 이해:**
        - **25분위 (하위 25%)**: 전체 단지 중 가격이 낮은 25% 그룹 (저가 단지)
        - **50분위 (중간)**: 중간 가격 단지
        - **75분위 (상위 25%)**: 가격이 높은 25% 그룹 (고가 단지)

        **그래프 해석:**
        - **막대 높이가 분위수마다 다름**: 그 요인의 영향이 가격대별로 다르다는 의미
          - 예: 면적 효과가 저가 단지에서 +8%, 고가 단지에서 +15% → 비싼 단지일수록 면적 프리미엄이 더 큼
        - **막대 높이가 모든 분위수에서 비슷**: 가격대 상관없이 일관된 영향

        **💡 팁:** 고가 단지에서 특히 큰 요인 = 부유층이 더 중요하게 여기는 특성.
        """)

    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_heterogeneity_months"))
    hetero_df = _get_heterogeneity(months)
    if hetero_df.empty:
        st.warning("분위수별 프리미엄을 계산할 데이터가 부족합니다.")
        return
    st.plotly_chart(build_heterogeneity_chart(hetero_df), width="stretch")
    st.dataframe(hetero_df, width="stretch", hide_index=True)


def render_complex_liquidity() -> None:
    st.header("🏘️ 거래 빈도와 유동성")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **유동성이란?** 얼마나 자주, 쉽게 거래되는지. 유동성이 높은 단지 = 사고팔기 쉬운 단지.

        **그래프 해석:**
        - **거래 빈도 버킷**: 연간 거래 발생 빈도에 따라 단지를 구간으로 나눔
          - 예: '월 0~0.5회', '월 0.5~1회', '월 1회 이상'
        - **구간별 가격 차이**: 거래가 잦은 단지가 가격도 높은지 보여줌
          - 거래가 활발한 단지 = 시장에서 선호하는 단지일 가능성
        - **유동성 계수 막대**: 거래 빈도가 1 단위 오를 때 가격이 몇 % 바뀌는지

        **💡 팁:** 거래가 너무 드문 단지는 급매물이 나와도 살 사람이 없어 유동성 리스크가 있습니다.
        """)

    result, bucket_df = _get_liquidity()
    if result.coefficients.empty and bucket_df.empty:
        st.warning("유동성 분석 데이터가 없습니다.")
        return
    if not bucket_df.empty:
        st.plotly_chart(build_liquidity_chart(bucket_df), width="stretch")
    if not result.coefficients.empty:
        _render_model_result("거래 유동성 구성 요소", result)
