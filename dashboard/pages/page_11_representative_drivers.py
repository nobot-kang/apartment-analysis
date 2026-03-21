"""Dashboard page for representative 59/84 price structure analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.representative_complex_analysis import (
    build_jeonse_ratio_band_chart,
    build_jeonse_ratio_band_frame,
    build_liquidity_chart,
    build_liquidity_frame,
    build_region_spread_frame,
    build_spread_chart,
    build_spread_ranking_frame,
    get_region_option_map,
)
from dashboard.data_loader import (
    load_representative_region_monthly,
    load_representative_rent_band_monthly,
    load_representative_trade_band_monthly,
)


def _select_region(region_options: dict[str, str], key: str, label: str = "지역") -> str | None:
    if not region_options:
        return None
    labels = list(region_options.keys())
    selected = st.selectbox(label, labels, index=0, key=key)
    return region_options[selected]


@st.cache_data(ttl=3600, show_spinner=False)
def _get_spread_frame(region_level: str, market_type: str) -> pd.DataFrame:
    return build_region_spread_frame(load_representative_region_monthly(), region_level=region_level, market_type=market_type)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_spread_ranking(region_level: str, market_type: str) -> pd.DataFrame:
    return build_spread_ranking_frame(_get_spread_frame(region_level, market_type))


@st.cache_data(ttl=3600, show_spinner=False)
def _get_spread_region_options(region_level: str, market_type: str) -> dict[str, str]:
    region_df = load_representative_region_monthly()
    filtered = region_df[(region_df["region_level"] == region_level) & (region_df["market_type"] == market_type)].copy()
    return get_region_option_map(filtered, region_level)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_jeonse_ratio(region_level: str, region_key: str) -> pd.DataFrame:
    return build_jeonse_ratio_band_frame(
        load_representative_trade_band_monthly(),
        load_representative_rent_band_monthly(),
        region_level=region_level,
        region_key=region_key,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_liquidity(region_level: str, area_band: int, region_key: str | None) -> pd.DataFrame:
    return build_liquidity_frame(
        load_representative_trade_band_monthly(),
        region_level=region_level,
        area_band=area_band,
        region_key=region_key,
    )


def _render_spread_section(*, market_type: str, title: str, key_prefix: str) -> None:
    st.header(title)
    controls = st.columns([1.0, 1.5])
    region_level = controls[0].radio(
        "지역 단위",
        options=["sigungu", "bjdong"],
        horizontal=True,
        format_func=lambda value: "시군구" if value == "sigungu" else "법정동",
        key=f"{key_prefix}_region_level",
    )
    region_key = _select_region(_get_spread_region_options(region_level, market_type), f"{key_prefix}_region_key")
    if region_key is None:
        st.warning("선택할 수 있는 지역이 없습니다.")
        return

    spread_df = _get_spread_frame(region_level, market_type)
    ranking_df = _get_spread_ranking(region_level, market_type)
    if spread_df.empty or ranking_df.empty:
        st.warning("선택한 시장의 spread 데이터가 없습니다.")
        return

    selected = spread_df[spread_df["region_code"].astype(str) == str(region_key)].sort_values("date")
    if selected.empty:
        st.warning("선택한 지역의 spread 데이터가 없습니다.")
        return

    latest = selected.iloc[-1]
    cols = st.columns(3)
    cols[0].metric("59형", f"{latest['price_59']:,.1f} 만원/평")
    cols[1].metric("84형", f"{latest['price_84']:,.1f} 만원/평")
    cols[2].metric("84/59 격차율", f"{latest['spread_ratio']:,.2f}%")
    st.plotly_chart(build_spread_chart(spread_df, str(region_key), f"{latest['region_name']} · {title}"), width="stretch")
    tables = st.columns(2)
    tables[0].dataframe(ranking_df.head(15), width="stretch", hide_index=True)
    tables[1].dataframe(ranking_df.tail(15).sort_values("spread_ratio"), width="stretch", hide_index=True)


def render_representative_sale_spread() -> None:
    _render_spread_section(market_type="sale", title="Representative Level 2 - 매매 84/59 spread", key_prefix="rep_sale_spread")


def render_representative_jeonse_spread() -> None:
    _render_spread_section(market_type="jeonse", title="Representative Level 2 - 전세 84/59 spread", key_prefix="rep_jeonse_spread")


def render_representative_wolse_spread() -> None:
    _render_spread_section(market_type="wolse", title="Representative Level 2 - 월세 84/59 spread", key_prefix="rep_wolse_spread")


def render_representative_jeonse_ratio() -> None:
    st.header("Representative Level 2 - 59형/84형 전세가율 비교")
    controls = st.columns([1.0, 1.5])
    region_level = controls[0].radio(
        "지역 단위",
        options=["sigungu", "bjdong"],
        horizontal=True,
        format_func=lambda value: "시군구" if value == "sigungu" else "법정동",
        key="rep_ratio_region_level",
    )
    region_key = _select_region(_get_spread_region_options(region_level, "sale"), "rep_ratio_region_key")
    if region_key is None:
        st.warning("선택할 수 있는 지역이 없습니다.")
        return

    ratio_df = _get_jeonse_ratio(region_level, region_key)
    if ratio_df.empty:
        st.warning("선택한 지역의 전세가율 데이터가 없습니다.")
        return

    latest = ratio_df.sort_values("date").iloc[-1]
    cols = st.columns(3)
    cols[0].metric("59형 전세가율", f"{latest.get('ratio_59', float('nan')):,.2f}%")
    cols[1].metric("84형 전세가율", f"{latest.get('ratio_84', float('nan')):,.2f}%")
    cols[2].metric("84-59 차이", f"{latest.get('gap_ratio', float('nan')):,.2f}pp")
    st.plotly_chart(build_jeonse_ratio_band_chart(ratio_df, "대표 평형 전세가율 비교"), width="stretch")
    st.dataframe(ratio_df.tail(24).sort_values("date", ascending=False), width="stretch", hide_index=True)


def render_representative_liquidity() -> None:
    st.header("Representative Level 2 - 거래발생률과 보간 의존도")
    controls = st.columns([1.0, 1.0, 1.5])
    region_level = controls[0].radio(
        "지역 단위",
        options=["sigungu", "bjdong"],
        horizontal=True,
        format_func=lambda value: "시군구" if value == "sigungu" else "법정동",
        key="rep_liquidity_region_level",
    )
    area_band = int(
        controls[1].radio(
            "대표 평형",
            options=[59, 84],
            horizontal=True,
            format_func=lambda value: f"{value}형",
            key="rep_liquidity_band",
        )
    )
    region_key = _select_region(_get_spread_region_options(region_level, "sale"), "rep_liquidity_region_key")
    if region_key is None:
        st.warning("선택할 수 있는 지역이 없습니다.")
        return

    liquidity_df = _get_liquidity(region_level, area_band, region_key)
    if liquidity_df.empty:
        st.warning("선택한 조건의 유동성 데이터가 없습니다.")
        return

    latest = liquidity_df.sort_values("date").iloc[-1]
    cols = st.columns(4)
    cols[0].metric("활성 단지", f"{int(latest['active_complexes']):,}")
    cols[1].metric("실관측 단지", f"{int(latest['observed_complexes']):,}")
    cols[2].metric("보간 비중", f"{latest['imputed_share']:,.1f}%")
    cols[3].metric("평균 fill age", f"{latest['avg_fill_age']:,.1f} 개월" if pd.notna(latest["avg_fill_age"]) else "N/A")
    st.plotly_chart(build_liquidity_chart(liquidity_df, f"{area_band}형 유동성"), width="stretch")
    st.dataframe(liquidity_df.tail(24).sort_values("date", ascending=False), width="stretch", hide_index=True)
