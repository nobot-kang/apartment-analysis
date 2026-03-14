"""Page 01 - Level 1 기초 현황."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import get_scope_codes
from analysis.level1 import (
    build_age_premium_chart,
    build_area_boxplot,
    build_jeonse_ratio_chart,
    build_monthly_volume_chart,
    build_ranking_animation,
    build_ranking_chart,
    load_age_premium,
    load_area_distribution,
    load_district_ranking,
    load_jeonse_ratio,
    load_monthly_volume,
)
from dashboard.data_loader import load_macro_monthly, load_rent_summary, load_trade_summary


@st.cache_data(ttl=3600)
def _get_monthly_volume(scope_name: str) -> pd.DataFrame:
    return load_monthly_volume(get_scope_codes(scope_name), scope_name)


@st.cache_data(ttl=3600)
def _get_ranking(year: int) -> pd.DataFrame:
    return load_district_ranking(year)


@st.cache_data(ttl=3600)
def _get_area_distribution(region_code: str, years: tuple[int, ...]) -> pd.DataFrame:
    return load_area_distribution([region_code], list(years))


@st.cache_data(ttl=3600)
def _get_age_premium(region_code: str, years: tuple[int, ...]) -> pd.DataFrame:
    return load_age_premium([region_code], list(years))


@st.cache_data(ttl=3600)
def _get_jeonse_ratio() -> pd.DataFrame:
    return load_jeonse_ratio()


def render() -> None:
    st.header("Level 1 - 기초 현황")

    trade_df = load_trade_summary()
    rent_df = load_rent_summary()
    macro_df = load_macro_monthly()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다. 집계 파이프라인을 먼저 실행해주세요.")
        return

    latest_ym = trade_df["ym"].max()
    latest_trade = trade_df[trade_df["ym"] == latest_ym]
    latest_rent = rent_df[(rent_df["ym"] == latest_ym) & (rent_df["rentType"] == "전세")]
    latest_ratio = (latest_rent["평균보증금"].mean() / latest_trade["평균거래금액"].mean() * 100) if not latest_rent.empty else float("nan")
    latest_rate = macro_df["bok_rate"].dropna().iloc[-1] if not macro_df.empty and "bok_rate" in macro_df.columns and not macro_df["bok_rate"].dropna().empty else float("nan")

    cols = st.columns(4)
    cols[0].metric("최근 평균 매매가", f"{latest_trade['평균거래금액'].mean():,.0f} 만원")
    cols[1].metric("최근 거래건수", f"{int(latest_trade['거래건수'].sum()):,} 건")
    cols[2].metric("평균 전세가율", f"{latest_ratio:.1f}%" if latest_ratio == latest_ratio else "N/A")
    cols[3].metric("한국 기준금리", f"{latest_rate:.2f}%" if latest_rate == latest_rate else "N/A")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["거래량 추이", "지역 랭킹", "면적 분포", "건축 연령", "전세가율"])

    with tab1:
        scope_name = st.selectbox("분석 범위", ["서울 전체", "경기 전체", "수도권 전체", *list(dict.fromkeys(trade_df["_region_name"].tolist()))], index=0)
        volume_df = _get_monthly_volume(scope_name)
        st.plotly_chart(build_monthly_volume_chart(volume_df, scope_name), width="stretch")

    with tab2:
        years = sorted(trade_df["year"].dropna().unique())
        selected_year = st.select_slider("랭킹 기준 연도", options=years, value=years[-1])
        metric = st.radio("랭킹 지표", ["avg_price", "avg_price_per_m2"], horizontal=True, format_func=lambda x: "평균 매매가" if x == "avg_price" else "평균 ㎡당 가격")
        ranking_df = _get_ranking(int(selected_year))
        st.plotly_chart(build_ranking_chart(ranking_df, int(selected_year), metric), width="stretch")
        with st.expander("연도별 랭킹 애니메이션", expanded=False):
            st.plotly_chart(build_ranking_animation(metric), width="stretch")

    with tab3:
        region_name = st.selectbox("면적 분포 지역", options=sorted(trade_df["_region_name"].unique()), index=0)
        region_code = trade_df.loc[trade_df["_region_name"] == region_name, "_lawd_cd"].iloc[0]
        year_choices = sorted(int(value) for value in trade_df["year"].dropna().unique())
        selected_years = st.multiselect("포함 연도", options=year_choices, default=year_choices[-4:])
        area_bin = st.radio("면적 구간", ["60㎡ 이하", "60~85㎡", "85㎡ 초과"], horizontal=True)
        if selected_years:
            area_df = _get_area_distribution(str(region_code), tuple(selected_years))
            st.plotly_chart(build_area_boxplot(area_df, area_bin, region_name), width="stretch")
        else:
            st.info("최소 1개 연도를 선택해주세요.")

    with tab4:
        region_name = st.selectbox("건축 연령 분석 지역", options=sorted(trade_df["_region_name"].unique()), index=min(1, len(trade_df["_region_name"].unique()) - 1))
        region_code = trade_df.loc[trade_df["_region_name"] == region_name, "_lawd_cd"].iloc[0]
        year_choices = sorted(int(value) for value in trade_df["year"].dropna().unique())
        selected_year = st.select_slider("건축 연령 기준 연도", options=year_choices, value=year_choices[-1])
        age_df = _get_age_premium(str(region_code), (int(selected_year),))
        st.plotly_chart(build_age_premium_chart(age_df, region_name, int(selected_year)), width="stretch")

    with tab5:
        ratio_df = _get_jeonse_ratio()
        region_name = st.selectbox("전세가율 지역", options=sorted(ratio_df["_region_name"].unique()) if not ratio_df.empty else [], index=0)
        if region_name:
            st.plotly_chart(build_jeonse_ratio_chart(ratio_df, region_name), width="stretch")

