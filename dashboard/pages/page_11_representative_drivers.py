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
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **Spread(격차)란?** 같은 시장(매매·전세·월세)에서 지역별로 84형과 59형 평당가의 차이 비율.

        **주요 수치 해석:**
        - **84/59 격차율 (%)**: 84형이 59형보다 평당 몇 % 더 비싼지.
          - 양수: 84형이 더 비쌈 (대형 선호 지역)
          - 음수 또는 0에 가까움: 두 평형 가격이 비슷함 (소형 희소성 또는 대형 할인)
        - **상위 15 지역**: 격차가 가장 큰 지역 = 84형이 59형보다 훨씬 비싼 곳.
        - **하위 15 지역**: 격차가 가장 작은 지역 = 두 평형 가격이 비슷한 곳.

        **💡 팁:** 격차가 큰 지역에서 59형을 사면 84형 대비 저평가일 수 있고, 격차가 줄어드는 추세라면 59형이 상대적으로 강세로 전환하는 신호일 수 있습니다.
        """)
    _render_spread_section(market_type="sale", title="🏡 매매 — 지역 간 평당가 격차", key_prefix="rep_sale_spread")


def render_representative_jeonse_spread() -> None:
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **전세 격차 분석.** 전세 보증금 기준으로 84형과 59형의 지역별 평당가 격차를 보여줍니다.

        **매매 격차와 다른 점:** 임차인(세입자) 관점에서의 평형 선호를 반영합니다. 전세에서 격차가 크면 임차인도 84형을 더 선호한다는 의미.

        **수치 해석:** 매매 격차 설명과 동일. 단, 가격은 전세 보증금 기준.
        """)
    _render_spread_section(market_type="jeonse", title="🏡 전세 — 지역 간 평당가 격차", key_prefix="rep_jeonse_spread")


def render_representative_wolse_spread() -> None:
    _render_spread_section(market_type="wolse", title="🏡 월세 — 지역 간 평당가 격차", key_prefix="rep_wolse_spread")


def render_representative_jeonse_ratio() -> None:
    st.header("🏡 59형·84형 전세가율 비교")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **전세가율 복습:** 전세 보증금 ÷ 매매가 × 100. 전세가 매매가의 몇 %인지.

        **59형 vs 84형 전세가율이 다른 이유:** 두 평형의 매매가 상승 속도와 전세 수요가 다를 수 있음.

        **주요 수치 해석:**
        - **59형 전세가율 (%)**: 59형의 전세 보증금이 매매가의 몇 %인지.
        - **84형 전세가율 (%)**: 84형의 전세 보증금이 매매가의 몇 %인지.
        - **84-59 차이 (pp)**: 두 평형 전세가율의 차이.
          - 예: +5pp → 84형이 59형보다 전세가율이 5%포인트 높음 (84형이 상대적으로 전세가 비쌈)
          - 마이너스 → 59형의 전세가율이 더 높음

        **⚠️ 전세가율 70% 이상이면:** 깡통 전세 위험 구간. 특히 주의하세요.
        """)

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
    st.header("🏡 실거래 빈도와 데이터 신뢰도")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이 페이지가 필요한 이유:** 거래가 드문 달에는 실제 거래 없이 앞뒤 데이터로 '추정'한 가격을 사용합니다. 이 페이지는 그 신뢰도를 확인하는 곳.

        **주요 수치 해석:**
        - **활성 단지**: 해당 달에 데이터가 있는 단지 (실거래 + 보간 포함).
        - **실관측 단지**: 실제 거래가 있어 직접 계산된 단지. 활성 단지 대비 비율이 낮으면 그 달은 거래가 드물었음.
        - **보간 비중 (%)**: 전체 데이터 중 추정값(보간)이 차지하는 비율.
          - 30% 이상: 그 달 데이터의 30%는 실거래 없이 추정된 것 → 주의 필요.
        - **평균 fill age (개월)**: 보간에 사용한 데이터가 평균 몇 달 전/후 것인지. 클수록 오래된 데이터로 채워진 것 → 정확도 낮음.

        **💡 팁:** 보간 비중이 높거나 fill age가 큰 구간의 가격 분석 결과는 신중하게 해석하세요.
        """)

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
