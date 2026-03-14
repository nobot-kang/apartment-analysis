"""Page 02 - Level 2 심화 비교 분석."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import get_scope_codes
from analysis.level2 import (
    build_conversion_rate_chart,
    build_district_year_heatmap,
    build_floor_premium_chart,
    build_volume_price_lag_chart,
    build_yoy_map,
    load_conversion_rate_data,
    load_floor_premium_data,
    load_volume_price_lag_data,
    load_yoy_map_data,
)
from dashboard.data_loader import load_trade_summary


@st.cache_data(ttl=3600)
def _get_floor_data(region_codes: tuple[str, ...], years: tuple[int, ...]) -> pd.DataFrame:
    return load_floor_premium_data(list(region_codes), list(years))


@st.cache_data(ttl=3600)
def _get_yoy_map(target_year: int) -> pd.DataFrame:
    return load_yoy_map_data(target_year)


@st.cache_data(ttl=3600)
def _get_lag_data(scope_name: str) -> pd.DataFrame:
    return load_volume_price_lag_data(get_scope_codes(scope_name), scope_name)


@st.cache_data(ttl=3600)
def _get_conversion_rate(scope_name: str) -> pd.DataFrame:
    return load_conversion_rate_data(get_scope_codes(scope_name), scope_name)


def render() -> None:
    st.header("Level 2 - 심화 비교 분석")

    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["지역 히트맵", "층수 프리미엄", "YoY 지도", "거래량-가격 선행"])

    with tab1:
        metric = st.radio(
            "히트맵 지표",
            ["avg_price", "avg_price_per_m2", "yoy_change"],
            horizontal=True,
            format_func=lambda value: {"avg_price": "평균 매매가", "avg_price_per_m2": "㎡당 가격", "yoy_change": "YoY 상승률"}[value],
        )
        st.plotly_chart(build_district_year_heatmap(metric), width="stretch")

    with tab2:
        region_options = sorted(trade_df["_region_name"].unique())
        selected_regions = st.multiselect("비교 지역", options=region_options, default=region_options[:3])
        selected_codes = tuple(trade_df[trade_df["_region_name"].isin(selected_regions)]["_lawd_cd"].astype(str).drop_duplicates().tolist())
        selected_year = int(st.select_slider("분석 연도", options=sorted(int(value) for value in trade_df["year"].dropna().unique()), value=int(trade_df["year"].max())))
        if selected_regions and selected_codes:
            floor_df = _get_floor_data(selected_codes, (selected_year,))
            st.plotly_chart(build_floor_premium_chart(floor_df, selected_regions, selected_year), width="stretch")
        else:
            st.info("최소 1개 지역을 선택해주세요.")

    with tab3:
        target_year = int(st.select_slider("지도 기준 연도", options=sorted(int(value) for value in trade_df["year"].dropna().unique()), value=int(trade_df["year"].max())))
        yoy_df = _get_yoy_map(target_year)
        st.caption("서울 자치구 중심점 버블맵으로 표시합니다. GeoJSON이 없더라도 대시보드가 동작하도록 구성했습니다.")
        st.plotly_chart(build_yoy_map(yoy_df, target_year), width="stretch")

    with tab4:
        scope_name = st.selectbox("선후행 분석 범위", ["서울 전체", "경기 전체", "수도권 전체", *sorted(trade_df["_region_name"].unique())], index=0)
        lag_df = _get_lag_data(scope_name)
        st.plotly_chart(build_volume_price_lag_chart(lag_df, scope_name), width="stretch")

        st.subheader("전월세 전환율")
        conversion_df = _get_conversion_rate(scope_name)
        st.plotly_chart(build_conversion_rate_chart(conversion_df, scope_name), width="stretch")

