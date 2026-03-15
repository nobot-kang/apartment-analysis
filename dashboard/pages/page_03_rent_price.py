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
    st.header("전월세 분석 - 전세/월세 추이")
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
    st.header("전월세 분석 - 보증금·월세 분포")
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