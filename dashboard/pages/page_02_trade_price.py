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
    st.header("📈 지역·연도 가격 지도")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 지표 선택 → ② 표의 색으로 지역·연도별 비교

        **주요 수치 해석:**
        - **색 진하기**: 짙은 빨강 = 해당 지역·연도에서 가장 높은 값. 짙은 파랑 = 가장 낮은 값.
        - **평균 매매가 (만원)**: 그 해 그 지역에서 거래된 아파트의 평균 가격.
        - **㎡당 가격 (만원/㎡)**: 면적을 표준화한 가격. 평수가 다른 지역을 공평하게 비교할 때 유용.
        - **전년 대비 상승률 (%)**: 전년도 같은 지역 대비 가격이 몇 % 올랐는지. 빨간 칸이 몇 년 연속 이어지면 지속적인 상승세.

        **💡 팁:** 세로(연도)로 읽으면 '특정 지역의 역사'가 보이고, 가로(지역)로 읽으면 '그 해 가장 비싼 지역'이 보여요.
        """)

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
    st.header("📈 층이 높을수록 얼마나 비쌀까?")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 비교할 지역 선택(복수 가능) → ② 기준 연도 선택 → ③ 층수별 가격 차이 확인

        **주요 수치 해석:**
        - **막대 값 (+%)**: 해당 층이 같은 단지 평균 대비 몇 % 더 비싸게 거래됐는지.
          - 예: 15층 +12% → 15층은 평균보다 12% 더 비쌈
          - 예: 1층 -8% → 1층은 평균보다 8% 더 쌈
        - **0% 기준선**: 이 선 위는 층 프리미엄, 아래는 층 할인.
        - **층수가 높을수록 꼭 비싼 건 아님**: 너무 높은 층은 엘리베이터 대기, 소방 문제 등으로 할인되는 경우도 있음.

        **💡 팁:** 지역마다 '층 프리미엄이 가장 잘 받히는 층수'가 달라요. 여러 지역을 한 번에 선택해서 비교해보세요.
        """)

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
    st.header("📈 전년 대비 가격 변화 지도")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 기준 연도 선택 → ② 지도에서 버블 색과 크기 확인

        **주요 수치 해석:**
        - **버블 색 (빨강)**: 전년 대비 가격이 오른 지역. 색이 짙을수록 많이 오름.
        - **버블 색 (파랑)**: 전년 대비 가격이 내린 지역.
        - **버블 크기**: 해당 지역의 거래량(거래가 많을수록 버블이 큼).
        - **전년 대비 변화율 (YoY %)**: 작년 같은 시기와 비교해 가격이 몇 % 바뀌었는지.
          - 예: +15% → 1년 전보다 15% 상승. 5억이었으면 지금 5.75억.

        **💡 팁:** 큰 버블 + 빨간색 = 거래도 많고 가격도 많이 오른 핫한 지역. 작은 버블 + 파란색 = 거래도 적고 가격도 빠지는 침체 지역.
        """)

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
    st.header("📈 거래량이 오르면 가격도 오를까?")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 분석 범위 선택 → ② 상관계수 막대그래프에서 가장 큰 막대 확인

        **핵심 개념 — 선행지표:** 거래량이 먼저 오르고, 그 후 몇 달 뒤 가격이 따라 오르는 경우가 많습니다. 거래량은 가격의 '예고편'입니다.

        **상관계수 (−1 ~ +1) 해석:**
        - **가장 큰 양수 막대**: 거래량이 오른 뒤 해당 개월 수 후에 가격이 가장 강하게 올랐다는 의미.
          - 예: 3개월 시차에서 +0.7 → 거래량 증가 3개월 후 가격 상승이 가장 뚜렷함.
        - **+0.7 이상**: 강한 양의 관계 (거래량↑ → 가격↑)
        - **0 근처**: 관계 없음
        - **마이너스**: 거래량이 올라도 가격이 오히려 내리는 비정상적 상황 (급매물 증가 등)

        **⚠️ 주의:** 상관관계는 인과관계가 아닙니다. "거래량이 올라서 가격이 올랐다"고 단정할 수는 없어요.
        """)

    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    scope_options = get_scope_option_list()
    scope_name = st.selectbox("선후행 분석 범위", scope_options, index=0, key="level2_lag_scope")
    lag_df = prepare_volume_price_lag(trade_df, get_scope_codes(scope_name), scope_name)
    st.plotly_chart(build_volume_price_lag_chart(lag_df, scope_name), width="stretch")


def render_conversion_rate() -> None:
    st.header("📈 전세를 월세로 바꾸면? (전환율)")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **전월세 전환율이란?** 전세 보증금을 월세로 바꿀 때 적용하는 연 이자율.

        **계산 예시:** 전환율 5%, 전세 보증금 3억
        - 월세 환산: 3억 × 5% ÷ 12개월 = 월 125만원

        **수치 해석:**
        - **전환율 높아짐**: 집주인에게 월세가 더 유리해지는 시기. 전세보다 월세 공급 증가 가능.
        - **전환율 낮아짐**: 세입자 입장에서 월세 부담이 줄어드는 시기.
        - **법정 한도**: 한국은 주택임대차보호법으로 전환율 상한을 규정합니다 (기준금리 + 일정 %p).

        **💡 팁:** 기준금리가 오르면 법정 전환율 상한도 올라가므로, 이 그래프의 추이와 금리 그래프를 함께 보면 좋습니다.
        """)

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