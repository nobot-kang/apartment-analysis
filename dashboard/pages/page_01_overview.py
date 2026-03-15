"""Page 01 - Level 1 기초 현황."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import get_scope_codes, load_trade_detail_df
from analysis.level1 import (
    build_age_premium_chart,
    build_area_boxplot,
    build_jeonse_ratio_chart,
    build_monthly_volume_chart,
    build_monthly_volume_frame,
    build_overview_metrics,
    build_ranking_animation,
    build_ranking_chart,
    filter_district_ranking,
    prepare_age_premium,
    prepare_area_distribution,
)
from dashboard.data_loader import (
    get_scope_option_list,
    load_district_year_metrics,
    load_jeonse_ratio_precomputed,
    load_macro_monthly,
    load_rent_summary,
    load_trade_summary,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_area_distribution(region_code: str, years: tuple[int, ...]):
    trade_detail = load_trade_detail_df(
        years=list(years),
        region_codes=[region_code],
        columns=["date", "price", "area", "dong_repr"],
    )
    return prepare_area_distribution(trade_detail)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_age_premium(region_code: str, year: int):
    trade_detail = load_trade_detail_df(
        years=[year],
        region_codes=[region_code],
        columns=["date", "price", "area", "age", "dong_repr"],
    )
    return prepare_age_premium(trade_detail)


def _render_level1_kpis() -> None:
    trade_df = load_trade_summary()
    rent_df = load_rent_summary()
    macro_df = load_macro_monthly()
    metrics = build_overview_metrics(trade_df, rent_df, macro_df)

    cols = st.columns(4)
    cols[0].metric("최근 평균 매매가", f"{metrics['latest_avg_trade']:,.0f} 만원" if metrics["latest_avg_trade"] == metrics["latest_avg_trade"] else "N/A")
    cols[1].metric("최근 거래건수", f"{int(metrics['latest_trade_count']):,} 건" if metrics["latest_trade_count"] == metrics["latest_trade_count"] else "N/A")
    cols[2].metric("평균 전세가율", f"{metrics['latest_ratio']:.1f}%" if metrics["latest_ratio"] == metrics["latest_ratio"] else "N/A")
    cols[3].metric("한국 기준금리", f"{metrics['latest_rate']:.2f}%" if metrics["latest_rate"] == metrics["latest_rate"] else "N/A")
    st.caption(f"기준월: {metrics['latest_ym']}")


def render_volume() -> None:
    st.header("Level 1 - 거래량 추이")
    trade_df = load_trade_summary()
    rent_df = load_rent_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다. 집계 파이프라인을 먼저 실행해주세요.")
        return

    _render_level1_kpis()
    scope_options = get_scope_option_list()
    scope_name = st.selectbox("분석 범위", scope_options, index=0, key="level1_volume_scope")
    volume_df = build_monthly_volume_frame(trade_df, rent_df, get_scope_codes(scope_name), scope_name)
    st.plotly_chart(build_monthly_volume_chart(volume_df, scope_name), width="stretch")


def render_ranking() -> None:
    st.header("Level 1 - 지역 랭킹")
    yearly_metrics = load_district_year_metrics()
    if yearly_metrics.empty:
        st.warning("연도별 지역 지표 데이터가 없습니다. 집계 파이프라인을 다시 실행해주세요.")
        return

    years = sorted(int(value) for value in yearly_metrics["year"].dropna().unique())
    selected_year = int(st.select_slider("랭킹 기준 연도", options=years, value=years[-1], key="level1_ranking_year"))
    metric = st.radio(
        "랭킹 지표",
        ["avg_price", "avg_price_per_m2"],
        horizontal=True,
        format_func=lambda value: "평균 매매가" if value == "avg_price" else "평균 ㎡당 가격",
        key="level1_ranking_metric",
    )
    ranking_df = filter_district_ranking(yearly_metrics, selected_year)
    st.plotly_chart(build_ranking_chart(ranking_df, selected_year, metric), width="stretch")


def render_ranking_animation() -> None:
    st.header("Level 1 - 랭킹 애니메이션")
    yearly_metrics = load_district_year_metrics()
    if yearly_metrics.empty:
        st.warning("연도별 지역 지표 데이터가 없습니다.")
        return

    metric = st.radio(
        "애니메이션 지표",
        ["avg_price", "avg_price_per_m2"],
        horizontal=True,
        format_func=lambda value: "평균 매매가" if value == "avg_price" else "평균 ㎡당 가격",
        key="level1_ranking_animation_metric",
    )
    st.plotly_chart(build_ranking_animation(yearly_metrics, metric), width="stretch")


def render_area_distribution() -> None:
    st.header("Level 1 - 면적 분포")
    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    region_options = sorted(trade_df["_region_name"].dropna().unique())
    region_name = st.selectbox("면적 분포 지역", options=region_options, index=0, key="level1_area_region")
    region_code = str(trade_df.loc[trade_df["_region_name"] == region_name, "_lawd_cd"].iloc[0])
    year_choices = sorted(int(value) for value in trade_df["year"].dropna().unique())
    default_years = tuple(year_choices[-4:] if len(year_choices) >= 4 else year_choices)
    selected_years = tuple(st.multiselect("포함 연도", options=year_choices, default=list(default_years), key="level1_area_years"))
    area_bin = st.radio("면적 구간", ["60㎡ 이하", "60~85㎡", "85㎡ 초과"], horizontal=True, key="level1_area_bin")

    if not selected_years:
        st.info("최소 1개 연도를 선택해주세요.")
        return

    area_df = _get_area_distribution(region_code, selected_years)
    st.plotly_chart(build_area_boxplot(area_df, area_bin, region_name), width="stretch")


def render_age_premium() -> None:
    st.header("Level 1 - 건축 연령 프리미엄")
    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    region_options = sorted(trade_df["_region_name"].dropna().unique())
    default_index = min(1, len(region_options) - 1)
    region_name = st.selectbox("건축 연령 분석 지역", options=region_options, index=default_index, key="level1_age_region")
    region_code = str(trade_df.loc[trade_df["_region_name"] == region_name, "_lawd_cd"].iloc[0])
    year_choices = sorted(int(value) for value in trade_df["year"].dropna().unique())
    selected_year = int(st.select_slider("건축 연령 기준 연도", options=year_choices, value=year_choices[-1], key="level1_age_year"))
    age_df = _get_age_premium(region_code, selected_year)
    st.plotly_chart(build_age_premium_chart(age_df, region_name, selected_year), width="stretch")


def render_jeonse_ratio() -> None:
    st.header("Level 1 - 전세가율")
    ratio_df = load_jeonse_ratio_precomputed()
    if ratio_df.empty:
        st.warning("선계산 전세가율 데이터가 없습니다. 집계 파이프라인을 다시 실행해주세요.")
        return

    region_options = sorted(ratio_df["_region_name"].dropna().unique())
    region_name = st.selectbox("전세가율 지역", options=region_options, index=0, key="level1_jeonse_region")
    st.plotly_chart(build_jeonse_ratio_chart(ratio_df, region_name), width="stretch")