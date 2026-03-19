"""Main entrypoint for the Streamlit dashboard."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from time import perf_counter

import streamlit as st
from loguru import logger

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

NAVIGATION = {
    "Level 1 · 기초 현황": {
        "거래량 추이": ("dashboard.pages.page_01_overview", "render_volume"),
        "지역 랭킹": ("dashboard.pages.page_01_overview", "render_ranking"),
        "랭킹 애니메이션": ("dashboard.pages.page_01_overview", "render_ranking_animation"),
        "면적 분포": ("dashboard.pages.page_01_overview", "render_area_distribution"),
        "건축 연령": ("dashboard.pages.page_01_overview", "render_age_premium"),
        "전세가율": ("dashboard.pages.page_01_overview", "render_jeonse_ratio"),
    },
    "Level 2 · 가격 비교": {
        "지역 히트맵": ("dashboard.pages.page_02_trade_price", "render_heatmap"),
        "층수 프리미엄": ("dashboard.pages.page_02_trade_price", "render_floor_premium"),
        "YoY 지도": ("dashboard.pages.page_02_trade_price", "render_yoy_map"),
        "거래량-가격 선행": ("dashboard.pages.page_02_trade_price", "render_lag_analysis"),
        "전월세 전환율": ("dashboard.pages.page_02_trade_price", "render_conversion_rate"),
    },
    "임대 시장": {
        "전세/월세 추이": ("dashboard.pages.page_03_rent_price", "render_rent_trend"),
        "보증금-월세 분포": ("dashboard.pages.page_03_rent_price", "render_deposit_rent_scatter"),
    },
    "Level 3 · 거시 연계": {
        "금리 시차": ("dashboard.pages.page_04_macro_indicators", "render_rate_lag"),
        "M2 관계": ("dashboard.pages.page_04_macro_indicators", "render_m2"),
        "환율 이벤트": ("dashboard.pages.page_04_macro_indicators", "render_fx_event"),
        "실질 가격": ("dashboard.pages.page_04_macro_indicators", "render_real_price"),
        "복합 상관": ("dashboard.pages.page_04_macro_indicators", "render_correlation"),
    },
    "Level 4 · 고급 분석": {
        "가격 예측": ("dashboard.pages.page_05_correlation", "render_prediction"),
        "군집 분석": ("dashboard.pages.page_05_correlation", "render_cluster"),
        "이상거래": ("dashboard.pages.page_05_correlation", "render_anomaly"),
        "정책 효과(DiD)": ("dashboard.pages.page_05_correlation", "render_did"),
        "시장 사이클": ("dashboard.pages.page_05_correlation", "render_cycle"),
    },
    "Complex Level 1 · 단지 특성": {
        "단지 프로파일링": ("dashboard.pages.page_06_complex_overview", "render_complex_profile"),
        "대단지 프리미엄": ("dashboard.pages.page_06_complex_overview", "render_complex_scale_premium"),
        "주차 프리미엄": ("dashboard.pages.page_06_complex_overview", "render_complex_parking_premium"),
        "밀도 프리미엄": ("dashboard.pages.page_06_complex_overview", "render_complex_density_premium"),
        "대지지분 프리미엄": ("dashboard.pages.page_06_complex_overview", "render_complex_land_premium"),
    },
    "Complex Level 2 · 가격 구성": {
        "매매 헤도닉": ("dashboard.pages.page_07_complex_drivers", "render_complex_sale_hedonic"),
        "전세 헤도닉": ("dashboard.pages.page_07_complex_drivers", "render_complex_jeonse_hedonic"),
        "월세 헤도닉": ("dashboard.pages.page_07_complex_drivers", "render_complex_wolse_hedonic"),
        "이질성 분석": ("dashboard.pages.page_07_complex_drivers", "render_complex_heterogeneity"),
        "유동성 분석": ("dashboard.pages.page_07_complex_drivers", "render_complex_liquidity"),
    },
    "Complex Level 3 · 동학/인과": {
        "롤링 계수": ("dashboard.pages.page_08_complex_dynamics", "render_complex_rolling_coefficients"),
        "패널 고정효과": ("dashboard.pages.page_08_complex_dynamics", "render_complex_panel_fe"),
        "거시 상호작용": ("dashboard.pages.page_08_complex_dynamics", "render_complex_macro_interactions"),
        "재건축 옵션": ("dashboard.pages.page_08_complex_dynamics", "render_complex_redevelopment"),
        "확산 효과": ("dashboard.pages.page_08_complex_dynamics", "render_complex_spillover"),
    },
    "Complex Level 4 · 예측/시나리오": {
        "매매가 예측": ("dashboard.pages.page_09_complex_forecast", "render_complex_sale_forecast"),
        "전세/월세 예측": ("dashboard.pages.page_09_complex_forecast", "render_complex_rent_forecast"),
        "12개월 수익률": ("dashboard.pages.page_09_complex_forecast", "render_complex_return_forecast"),
        "비율 예측": ("dashboard.pages.page_09_complex_forecast", "render_complex_ratio_forecast"),
        "시나리오": ("dashboard.pages.page_09_complex_forecast", "render_complex_scenario"),
    },
}


def _resolve_renderer(module_name: str, function_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main() -> None:
    st.set_page_config(
        page_title="부동산 시계열 분석 대시보드",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("부동산 시계열 분석")
    st.sidebar.markdown("섹션 단위 lazy loading 으로 필요한 분석만 렌더링합니다.")

    group_name = st.sidebar.radio("분석 그룹", list(NAVIGATION.keys()))
    section_name = st.sidebar.radio("분석 섹션", list(NAVIGATION[group_name].keys()))
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
