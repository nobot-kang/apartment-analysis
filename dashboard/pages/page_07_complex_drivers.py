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
    st.header("Complex Level 2 - 매매 헤도닉 모형")
    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_sale_hedonic_months"))
    result = _get_sale_hedonic(months)
    if result.coefficients.empty:
        st.warning("매매 헤도닉 모형을 추정할 데이터가 부족합니다.")
        return
    _render_model_result("매매 가격 구성 요소", result)


def render_complex_jeonse_hedonic() -> None:
    st.header("Complex Level 2 - 전세 헤도닉 모형")
    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_jeonse_hedonic_months"))
    result = _get_jeonse_hedonic(months)
    if result.coefficients.empty:
        st.warning("전세 헤도닉 모형을 추정할 데이터가 부족합니다.")
        return
    _render_model_result("전세 가격 구성 요소", result)


def render_complex_wolse_hedonic() -> None:
    st.header("Complex Level 2 - 월세 헤도닉 모형")
    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_wolse_hedonic_months"))
    result = _get_wolse_hedonic(months)
    if result.coefficients.empty:
        st.warning("월세 헤도닉 모형을 추정할 데이터가 부족합니다.")
        return
    _render_model_result("월세 가격 구성 요소", result)


def render_complex_heterogeneity() -> None:
    st.header("Complex Level 2 - 분위수별 이질성 분석")
    months = int(st.select_slider("스냅샷 기간", options=[6, 12, 24], value=12, key="complex_heterogeneity_months"))
    hetero_df = _get_heterogeneity(months)
    if hetero_df.empty:
        st.warning("분위수별 프리미엄을 계산할 데이터가 부족합니다.")
        return
    st.plotly_chart(build_heterogeneity_chart(hetero_df), width="stretch")
    st.dataframe(hetero_df, width="stretch", hide_index=True)


def render_complex_liquidity() -> None:
    st.header("Complex Level 2 - 유동성 분석")
    result, bucket_df = _get_liquidity()
    if result.coefficients.empty and bucket_df.empty:
        st.warning("유동성 분석 데이터가 없습니다.")
        return
    if not bucket_df.empty:
        st.plotly_chart(build_liquidity_chart(bucket_df), width="stretch")
    if not result.coefficients.empty:
        _render_model_result("거래 유동성 구성 요소", result)
