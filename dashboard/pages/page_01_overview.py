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
    st.header("📍 거래량과 가격 흐름")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 상단 요약 카드(현재 시장 상태 파악) → ② 그래프(시간 흐름 확인)

        **주요 수치 해석:**
        - **최근 평균 매매가 (만원)**: 1만원 = 10,000원. 예: 80,000만원 = 8억 원.
        - **최근 거래건수 (건)**: 숫자가 클수록 시장이 활발한 상태. 거래가 줄면 매수자·매도자가 눈치를 보는 신호.
        - **평균 전세가율 (%)**: 전세 보증금이 매매가의 몇 %인지. 80% 이상이면 깡통 전세 위험 주의.
        - **기준금리 (%)**: 한국은행이 정하는 기준 이자율. 금리가 오르면 대출이 비싸져 집값 하락 압력이 생김.
        - **그래프 선**: 선이 위로 올라가면 해당 월에 거래량(또는 가격)이 증가한 것.

        **⚠️ 주의:** 거래량이 갑자기 줄어도 계절적 요인(명절, 연말)일 수 있으니 1~2개월 연속 추이를 보세요.
        """)
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
    st.header("📍 지역별 가격 순위")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 연도 슬라이더로 기준 연도 선택 → ② 지표 선택 → ③ 막대그래프로 순위 확인

        **주요 수치 해석:**
        - **평균 매매가 (만원)**: 해당 지역에서 그 해에 거래된 아파트의 평균 가격. 막대가 길수록 비싼 지역.
        - **평균 ㎡당 가격 (만원/㎡)**: 면적을 맞춰 비교한 가격. 큰 평수가 많은 지역과 작은 평수 지역을 공평하게 비교할 때 유용.
        - **막대 길이**: 긴 막대 = 해당 지역이 그 해 가장 비싸게 거래된 지역.

        **💡 팁:** 연도를 바꿔가며 보면 '어느 지역이 언제부터 상승하기 시작했는지' 파악할 수 있어요.
        """)

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
    st.header("📍 순위 변화 애니메이션")
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
    st.header("📍 아파트 크기별 가격")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 지역 선택 → ② 연도 선택 → ③ 면적 구간 선택 → ④ 박스플롯 해석

        **주요 수치 해석:**
        - **박스(상자) 중간 선**: 해당 크기 아파트의 거래 가격 중간값(절반은 이보다 싸고, 절반은 이보다 비쌈).
        - **상자 위아래 경계**: 전체 거래의 중간 50% 가격 범위. 상자가 클수록 가격 편차가 크다는 뜻.
        - **상자 위아래로 뻗은 선(수염)**: 일반적인 가격 범위의 최대·최소.
        - **점(튀어나온 값)**: 평균에서 크게 벗어난 특이 거래. 재건축 단지나 특수 물건일 가능성.

        **면적 구간 안내:** 60㎡ 이하 ≈ 18평 이하 (소형), 60~85㎡ ≈ 18~26평 (중형), 85㎡ 초과 ≈ 26평 이상 (대형)
        """)

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
    st.header("📍 신축 vs 구축 — 연식 프리미엄")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 지역 선택 → ② 기준 연도 선택 → ③ 그래프에서 연식별 가격 차이 확인

        **주요 수치 해석:**
        - **프리미엄 (%)**: 해당 연식 아파트가 같은 지역 평균 대비 몇 % 더 비싸거나 싼지.
          - 예: +15% → 그 연식 아파트는 평균보다 15% 더 비쌈 (신축 프리미엄)
          - 예: -10% → 그 연식 아파트는 평균보다 10% 더 쌈 (구축 할인)
        - **0% 기준선**: 이 선을 기준으로 위는 프리미엄(비쌈), 아래는 할인(쌈).
        - **연식 범위**: 건축한 지 얼마나 됐는지 (예: 5년 이하 = 신축, 20년 초과 = 구축).

        **💡 팁:** 지역마다 신축 프리미엄 크기가 다릅니다. 재건축 이슈가 있는 지역은 오히려 구축이 비쌀 수도 있어요.
        """)

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
    st.header("📍 전세가율 현황")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **전세가율이란?** 전세 보증금 ÷ 매매가 × 100. 전세가 매매가의 몇 %인지 나타내는 숫자.

        **수치 해석 가이드:**
        - **~50%**: 전세가 매매가보다 훨씬 낮아 안전한 편.
        - **60~70%**: 일반적인 수준.
        - **80% 이상**: 전세 보증금이 매매가에 근접 → 집값이 조금만 내려도 보증금을 못 돌려받을 위험(깡통 전세).
        - **선이 올라갈 때**: 집값 대비 전세가 오르는 중. 전세 수요가 강하거나 집값이 내리는 신호.
        - **선이 내려갈 때**: 집값이 전세보다 더 빠르게 오르는 중.

        **⚠️ 주의:** 전세가율이 갑자기 급등하면 해당 지역 전세 사기 위험이 높아질 수 있습니다.
        """)

    ratio_df = load_jeonse_ratio_precomputed()
    if ratio_df.empty:
        st.warning("선계산 전세가율 데이터가 없습니다. 집계 파이프라인을 다시 실행해주세요.")
        return

    region_options = sorted(ratio_df["_region_name"].dropna().unique())
    region_name = st.selectbox("전세가율 지역", options=region_options, index=0, key="level1_jeonse_region")
    st.plotly_chart(build_jeonse_ratio_chart(ratio_df, region_name), width="stretch")