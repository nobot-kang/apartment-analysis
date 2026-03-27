"""Main entrypoint for the Streamlit dashboard."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable

import streamlit as st
from loguru import logger

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

NAVIGATION = {
    "📍 시장 한눈에 보기": {
        "데이터 진단 & 시장 스냅샷": ("dashboard.pages.page_00_market_snapshot_diagnostics", "render_snapshot"),
        "취소·직거래 비율 진단": ("dashboard.pages.page_14_trade_filter_diagnostics", "render_trade_filter_diagnostics"),
        "거래량과 가격 흐름": ("dashboard.pages.page_01_overview", "render_volume"),
        "지역별 가격 순위": ("dashboard.pages.page_01_overview", "render_ranking"),
        "순위 변화 애니메이션": ("dashboard.pages.page_01_overview", "render_ranking_animation"),
        "아파트 크기별 가격": ("dashboard.pages.page_01_overview", "render_area_distribution"),
        "신축 vs 구축 — 연식 프리미엄": ("dashboard.pages.page_01_overview", "render_age_premium"),
        "전세가율 현황": ("dashboard.pages.page_01_overview", "render_jeonse_ratio"),
        "전세·월세 거래 흐름": ("dashboard.pages.page_03_rent_price", "render_rent_trend"),
        "보증금과 월세의 관계": ("dashboard.pages.page_03_rent_price", "render_deposit_rent_scatter"),
    },
    "📈 가격 변화 깊이 보기": {
        "지역·연도 가격 지도": ("dashboard.pages.page_02_trade_price", "render_heatmap"),
        "층이 높을수록 얼마나 비쌀까?": ("dashboard.pages.page_02_trade_price", "render_floor_premium"),
        "전년 대비 가격 변화 지도": ("dashboard.pages.page_02_trade_price", "render_yoy_map"),
        "거래량이 오르면 가격도 오를까?": ("dashboard.pages.page_02_trade_price", "render_lag_analysis"),
        "전세를 월세로 바꾸면? (전환율)": ("dashboard.pages.page_02_trade_price", "render_conversion_rate"),
    },
    "🌍 경제 환경과 부동산": {
        "금리가 오르면 집값은 언제 내릴까?": ("dashboard.pages.page_04_macro_indicators", "render_rate_lag"),
        "돈이 많이 풀리면 집값이 오를까? (통화량)": ("dashboard.pages.page_04_macro_indicators", "render_m2"),
        "환율 급변 때 아파트는?": ("dashboard.pages.page_04_macro_indicators", "render_fx_event"),
        "물가 반영 후 실제 집값 변화": ("dashboard.pages.page_04_macro_indicators", "render_real_price"),
        "경제 지표들과 집값 — 상관관계 종합": ("dashboard.pages.page_04_macro_indicators", "render_correlation"),
    },
    "🔬 심화 분석": {
        "AI 가격 예측 모델": ("dashboard.pages.page_05_correlation", "render_prediction"),
        "비슷한 가격 패턴 지역 묶기": ("dashboard.pages.page_05_correlation", "render_cluster"),
        "수상한 거래 탐지": ("dashboard.pages.page_05_correlation", "render_anomaly"),
        "정책 시행 전후 비교": ("dashboard.pages.page_05_correlation", "render_did"),
        "지금 시장은 어느 단계일까?": ("dashboard.pages.page_05_correlation", "render_cycle"),
    },
    "🏘️ 단지 분석 — ① 단지 기본 정보": {
        "단지 규모·특성 프로필": ("dashboard.pages.page_06_complex_overview", "render_complex_profile"),
        "대단지일수록 얼마나 비쌀까?": ("dashboard.pages.page_06_complex_overview", "render_complex_scale_premium"),
        "주차 공간이 가격에 미치는 영향": ("dashboard.pages.page_06_complex_overview", "render_complex_parking_premium"),
        "건물 밀도와 가격의 관계": ("dashboard.pages.page_06_complex_overview", "render_complex_density_premium"),
        "세대당 땅 넓이와 가격의 관계": ("dashboard.pages.page_06_complex_overview", "render_complex_land_premium"),
    },
    "🏘️ 단지 분석 — ② 가격을 결정하는 요인": {
        "매매가 구성 요인 분석": ("dashboard.pages.page_07_complex_drivers", "render_complex_sale_hedonic"),
        "전세가 구성 요인 분석": ("dashboard.pages.page_07_complex_drivers", "render_complex_jeonse_hedonic"),
        "월세 구성 요인 분석": ("dashboard.pages.page_07_complex_drivers", "render_complex_wolse_hedonic"),
        "가격대별 반응 차이 (싼 단지 vs 비싼 단지)": ("dashboard.pages.page_07_complex_drivers", "render_complex_heterogeneity"),
        "거래 빈도와 유동성": ("dashboard.pages.page_07_complex_drivers", "render_complex_liquidity"),
    },
    "🏘️ 단지 분석 — ③ 시간에 따른 가격 변화": {
        "시간에 따라 변하는 가격 영향력": ("dashboard.pages.page_08_complex_dynamics", "render_complex_rolling_coefficients"),
        "단지 고유 특성 제거 후 순수 시장 효과": ("dashboard.pages.page_08_complex_dynamics", "render_complex_panel_fe"),
        "경기 국면별 단지 가격 반응": ("dashboard.pages.page_08_complex_dynamics", "render_complex_macro_interactions"),
        "재건축 기대감이 가격에 미치는 영향": ("dashboard.pages.page_08_complex_dynamics", "render_complex_redevelopment"),
        "인근 단지 가격 상승이 퍼지는 속도": ("dashboard.pages.page_08_complex_dynamics", "render_complex_spillover"),
    },
    "🏘️ 단지 분석 — ④ 미래 가격 예측": {
        "매매가 예측": ("dashboard.pages.page_09_complex_forecast", "render_complex_sale_forecast"),
        "전세·월세 예측": ("dashboard.pages.page_09_complex_forecast", "render_complex_rent_forecast"),
        "12개월 수익률 예측": ("dashboard.pages.page_09_complex_forecast", "render_complex_return_forecast"),
        "전세가율·전환율 예측": ("dashboard.pages.page_09_complex_forecast", "render_complex_ratio_forecast"),
        "경기 변화 시나리오 시뮬레이션": ("dashboard.pages.page_09_complex_forecast", "render_complex_scenario"),
    },
    "🏡 대표 단지 비교 — ① 전체 현황": {
        "대표 단지 데이터 커버리지": ("dashboard.pages.page_10_representative_overview", "render_representative_coverage"),
        "지역별 평당가 흐름": ("dashboard.pages.page_10_representative_overview", "render_representative_region_timeline"),
        "59형 vs 84형 가격 비교": ("dashboard.pages.page_10_representative_overview", "render_representative_band_comparison"),
        "특정 시점 지역별 가격 분포": ("dashboard.pages.page_10_representative_overview", "render_representative_distribution_snapshot"),
        "단지별 59형·84형 가격 차이 추이": ("dashboard.pages.page_10_representative_overview", "render_representative_pair_gap_history"),
    },
    "🏡 대표 단지 비교 — ② 가격 격차 구조": {
        "매매 — 지역 간 평당가 격차": ("dashboard.pages.page_11_representative_drivers", "render_representative_sale_spread"),
        "전세 — 지역 간 평당가 격차": ("dashboard.pages.page_11_representative_drivers", "render_representative_jeonse_spread"),
        "월세 — 지역 간 평당가 격차": ("dashboard.pages.page_11_representative_drivers", "render_representative_wolse_spread"),
        "59형·84형 전세가율 비교": ("dashboard.pages.page_11_representative_drivers", "render_representative_jeonse_ratio"),
        "실거래 빈도와 데이터 신뢰도": ("dashboard.pages.page_11_representative_drivers", "render_representative_liquidity"),
    },
    "🏡 대표 단지 비교 — ③ 가격 격차 변화 원인": {
        "시간에 따라 변하는 평형 가격 영향력": ("dashboard.pages.page_12_representative_dynamics", "render_representative_rolling_coefficients"),
        "단지 특성 제거 후 84형·59형 격차 변화": ("dashboard.pages.page_12_representative_dynamics", "render_representative_panel_fe"),
        "경기 국면별 평형 격차 반응": ("dashboard.pages.page_12_representative_dynamics", "render_representative_regime_response"),
        "지역 간 가격 격차 전파": ("dashboard.pages.page_12_representative_dynamics", "render_representative_spillover"),
        "가격 격차가 평균으로 돌아오는 경향": ("dashboard.pages.page_12_representative_dynamics", "render_representative_mean_reversion"),
    },
    "🏡 대표 단지 비교 — ④ 예측과 투자 판단": {
        "59형 평당가 수익률 예측": ("dashboard.pages.page_13_representative_forecast", "render_representative_sale59_forecast"),
        "84형 평당가 수익률 예측": ("dashboard.pages.page_13_representative_forecast", "render_representative_sale84_forecast"),
        "84형·59형 가격 차이 예측": ("dashboard.pages.page_13_representative_forecast", "render_representative_gap_forecast"),
        "관심 지역·단지 순위 스크리닝": ("dashboard.pages.page_13_representative_forecast", "render_representative_screening"),
        "경기 변화 시나리오 시뮬레이션": ("dashboard.pages.page_13_representative_forecast", "render_representative_scenario"),
    },
    
}


def _resolve_renderer(module_name: str, function_name: str) -> Callable[[], None]:
    """Dynamically resolve the selected dashboard renderer function."""
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="부동산 시계열 분석 대시보드",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🏠 서울 아파트 시장 분석")
    st.sidebar.markdown("원하는 분석만 선택해서 볼 수 있습니다.")

    group_name = st.sidebar.radio("분석 카테고리", list(NAVIGATION.keys()))
    section_name = st.sidebar.radio("세부 항목", list(NAVIGATION[group_name].keys()))
    module_name, function_name = NAVIGATION[group_name][section_name]

    start = perf_counter()
    with st.spinner(f"{section_name} 로딩 중..."):
        render = _resolve_renderer(module_name, function_name)
        render()
    elapsed = perf_counter() - start

    logger.info(
        "dashboard render complete | group={} | section={} | elapsed={:.3f}s",
        group_name,
        section_name,
        elapsed,
    )
    st.sidebar.metric("마지막 로딩", f"{elapsed:.2f}초")


if __name__ == "__main__":
    main()
