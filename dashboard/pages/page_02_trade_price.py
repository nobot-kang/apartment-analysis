"""Page 02 - Level 2 심화 비교 분석."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import get_scope_codes, load_trade_detail_df
from analysis.level2 import (
    build_conversion_rate_chart,
    build_district_year_heatmap,
    build_floor_premium_chart,
    build_volume_price_lag_chart,
    build_yoy_map,
    filter_conversion_rate,
    prepare_floor_premium,
    prepare_volume_price_lag,
    prepare_yoy_map,
)
from dashboard.data_loader import (
    get_scope_option_list,
    load_conversion_rate_precomputed,
    load_district_year_metrics,
    load_trade_summary,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_floor_data(region_codes: tuple[str, ...], year: int):
    trade_detail = load_trade_detail_df(
        years=[year],
        region_codes=list(region_codes),
        columns=["date", "price", "area", "floor", "dong_repr"],
    )
    return prepare_floor_premium(trade_detail)


def render_heatmap() -> None:
    st.header("Level 2 - 지역 히트맵")
    yearly_metrics = load_district_year_metrics()
    if yearly_metrics.empty:
        st.warning("연도별 지역 지표 데이터가 없습니다.")
        return

    metric = st.radio(
        "히트맵 지표",
        ["avg_price", "avg_price_per_m2", "yoy_change"],
        horizontal=True,
        format_func=lambda value: {"avg_price": "평균 매매가", "avg_price_per_m2": "㎡당 가격", "yoy_change": "YoY 상승률"}[value],
        key="level2_heatmap_metric",
    )
    st.plotly_chart(build_district_year_heatmap(yearly_metrics, metric), width="stretch")


def render_floor_premium() -> None:
    st.header("Level 2 - 층수 프리미엄")
    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    region_options = sorted(trade_df["_region_name"].dropna().unique())
    selected_regions = st.multiselect("비교 지역", options=region_options, default=region_options[:3], key="level2_floor_regions")
    selected_codes = tuple(
        trade_df[trade_df["_region_name"].isin(selected_regions)]["_lawd_cd"].astype(str).drop_duplicates().tolist()
    )
    selected_year = int(
        st.select_slider(
            "분석 연도",
            options=sorted(int(value) for value in trade_df["year"].dropna().unique()),
            value=int(trade_df["year"].max()),
            key="level2_floor_year",
        )
    )

    if not selected_regions or not selected_codes:
        st.info("최소 1개 지역을 선택해주세요.")
        return

    floor_df = _get_floor_data(selected_codes, selected_year)
    st.plotly_chart(build_floor_premium_chart(floor_df, selected_regions, selected_year), width="stretch")


def render_yoy_map() -> None:
    st.header("Level 2 - YoY 지도")
    yearly_metrics = load_district_year_metrics()
    if yearly_metrics.empty:
        st.warning("연도별 지역 지표 데이터가 없습니다.")
        return

    years = sorted(int(value) for value in yearly_metrics["year"].dropna().unique())
    target_year = int(st.select_slider("지도 기준 연도", options=years, value=years[-1], key="level2_yoy_year"))
    yoy_df = prepare_yoy_map(yearly_metrics, target_year)
    st.caption("서울 자치구 중심점 버블맵으로 표시합니다. GeoJSON 없이 빠르게 렌더되도록 구성했습니다.")
    st.plotly_chart(build_yoy_map(yoy_df, target_year), width="stretch")


def render_lag_analysis() -> None:
    st.header("Level 2 - 거래량-가격 선행 분석")
    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    scope_options = get_scope_option_list()
    scope_name = st.selectbox("선후행 분석 범위", scope_options, index=0, key="level2_lag_scope")
    lag_df = prepare_volume_price_lag(trade_df, get_scope_codes(scope_name), scope_name)
    st.plotly_chart(build_volume_price_lag_chart(lag_df, scope_name), width="stretch")


def render_conversion_rate() -> None:
    st.header("Level 2 - 전월세 전환율")
    conversion_df = load_conversion_rate_precomputed()
    if conversion_df.empty:
        st.warning("선계산 전월세 전환율 데이터가 없습니다.")
        return

    available_scopes = sorted(conversion_df["scope_name"].dropna().unique())
    default_options = [scope for scope in get_scope_option_list() if scope in set(available_scopes)]
    scope_name = st.selectbox("전환율 범위", default_options or available_scopes, index=0, key="level2_conversion_scope")
    scope_df = filter_conversion_rate(conversion_df, scope_name)
    st.plotly_chart(build_conversion_rate_chart(scope_df, scope_name), width="stretch")
    if not scope_df.empty:
        latest = scope_df.sort_values("date").iloc[-1]
        st.metric("최근 전환율", f"{latest['conversion_rate']:.2f}%")