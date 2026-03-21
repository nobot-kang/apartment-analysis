"""Page 03 - 전월세 심화 분석."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import aggregate_rent_scope, get_scope_codes, load_rent_detail_df
from dashboard.data_loader import get_scope_option_list, load_rent_summary


@st.cache_data(ttl=3600, show_spinner=False)
def _get_rent_scope(scope_name: str):
    rent_df = load_rent_summary()
    return aggregate_rent_scope(rent_df, get_scope_codes(scope_name), scope_name)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_wolse_detail(region_codes: tuple[str, ...], years: tuple[int, ...]):
    return load_rent_detail_df(
        years=list(years),
        region_codes=list(region_codes),
        columns=["date", "deposit", "monthly_rent", "rentType", "dong_repr"],
    )


def render_rent_trend() -> None:
    st.header("📍 전세·월세 거래 흐름")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 분석 범위 선택 → ② 보증금 추이 그래프 → ③ 거래건수 그래프

        **주요 수치 해석:**
        - **평균 보증금 (만원)**: 해당 월 전세 또는 월세 계약의 평균 보증금. 선이 오르면 세입자 부담 증가.
        - **거래건수 (건)**: 전세와 월세 각각 몇 건이 계약됐는지. 전세 비중이 줄고 월세가 늘면 집주인이 월세를 선호하는 시장으로 전환 중.
        - **전세 vs 월세 색**: 두 선의 간격이 벌어지면 전세와 월세 시장이 서로 다른 방향으로 움직이는 것.

        **💡 팁:** 전세 거래가 급감하고 월세가 급증하는 시점은 전세 대출 규제나 금리 인상과 맞닿아 있는 경우가 많아요.
        """)

    rent_df = load_rent_summary()
    if rent_df.empty:
        st.warning("전월세 집계 데이터가 없습니다.")
        return

    scope_options = get_scope_option_list()
    scope_name = st.selectbox("분석 범위", scope_options, index=0, key="rent_trend_scope")
    scope_df = _get_rent_scope(scope_name)
    if scope_df.empty:
        st.info("선택 범위의 전월세 데이터가 없습니다.")
        return

    fig_deposit = px.line(
        scope_df,
        x="date",
        y="평균보증금",
        color="rentType",
        title=f"{scope_name} 평균 보증금 추이",
        labels={"date": "월", "평균보증금": "평균 보증금 (만원)", "rentType": "유형"},
    )
    st.plotly_chart(fig_deposit, width="stretch")

    fig_count = px.bar(
        scope_df,
        x="date",
        y="거래건수",
        color="rentType",
        barmode="group",
        title=f"{scope_name} 전세/월세 거래건수",
        labels={"date": "월", "거래건수": "거래건수", "rentType": "유형"},
    )
    st.plotly_chart(fig_count, width="stretch")


def render_deposit_rent_scatter() -> None:
    st.header("📍 보증금과 월세의 관계")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 분석 범위·연도 선택 → ② 산점도에서 점들의 분포 확인

        **주요 수치 해석:**
        - **각 점 하나**: 실제 월세 계약 1건. X축=보증금, Y축=월세.
        - **오른쪽 위**: 보증금도 높고 월세도 높은 고가 매물.
        - **왼쪽 아래**: 보증금도 낮고 월세도 낮은 저가 매물.
        - **점들의 기울기**: 오른쪽 위로 향하는 추세선이 가파를수록 보증금과 월세가 함께 움직임.
        - **연도별 색 구분**: 특정 연도 점들이 전반적으로 오른쪽 위로 이동하면 그 해에 가격이 올랐다는 의미.

        **💡 팁:** 보증금이 매우 낮으면서 월세가 높은 점(왼쪽 위)은 집주인이 현금 흐름을 중시하는 조건. 반대로 보증금이 높고 월세가 낮은 점(오른쪽 아래)은 세입자가 목돈을 맡기는 조건.
        """)

    rent_df = load_rent_summary()
    if rent_df.empty:
        st.warning("전월세 집계 데이터가 없습니다.")
        return

    scope_options = get_scope_option_list()
    scope_name = st.selectbox("분석 범위", scope_options, index=0, key="rent_scatter_scope")
    region_codes = tuple(get_scope_codes(scope_name))
    year_choices = sorted(int(value) for value in rent_df["year"].dropna().unique())
    default_years = tuple(year_choices[-3:] if len(year_choices) >= 3 else year_choices)
    selected_years = tuple(st.multiselect("포함 연도", options=year_choices, default=list(default_years), key="rent_scatter_years"))
    if not selected_years:
        st.info("최소 1개 연도를 선택해주세요.")
        return

    detail_df = _get_wolse_detail(region_codes, selected_years)
    wolse = detail_df[detail_df["rentType"] == "월세"].copy() if not detail_df.empty else detail_df
    if wolse.empty:
        st.info("선택 범위의 월세 상세 데이터가 없습니다.")
        return

    fig_scatter = px.scatter(
        wolse,
        x="deposit",
        y="monthly_rent",
        color="year",
        title=f"{scope_name} 보증금-월세 분포",
        labels={"deposit": "보증금 (만원)", "monthly_rent": "월세 (만원)", "year": "연도"},
        opacity=0.55,
    )
    st.plotly_chart(fig_scatter, width="stretch")