"""Streamlit 대시보드 메인 앱."""

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
    "Level 2 · 심화 비교": {
        "지역 히트맵": ("dashboard.pages.page_02_trade_price", "render_heatmap"),
        "층수 프리미엄": ("dashboard.pages.page_02_trade_price", "render_floor_premium"),
        "YoY 지도": ("dashboard.pages.page_02_trade_price", "render_yoy_map"),
        "거래량-가격 선행": ("dashboard.pages.page_02_trade_price", "render_lag_analysis"),
        "전월세 전환율": ("dashboard.pages.page_02_trade_price", "render_conversion_rate"),
    },
    "전월세 분석": {
        "전세/월세 추이": ("dashboard.pages.page_03_rent_price", "render_rent_trend"),
        "보증금·월세 분포": ("dashboard.pages.page_03_rent_price", "render_deposit_rent_scatter"),
    },
    "Level 3 · 거시 연계": {
        "금리 시차 상관": ("dashboard.pages.page_04_macro_indicators", "render_rate_lag"),
        "M2 연계": ("dashboard.pages.page_04_macro_indicators", "render_m2"),
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
}


def _resolve_renderer(module_name: str, function_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main() -> None:
    """Streamlit 앱 진입점."""
    st.set_page_config(
        page_title="부동산 실거래가 분석 대시보드",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("부동산 실거래가 분석")
    st.sidebar.markdown("5초 안쪽 탐색을 목표로 분석을 섹션 단위로 분리했습니다.")

    group_name = st.sidebar.radio("분석 그룹", list(NAVIGATION.keys()))
    section_name = st.sidebar.radio("분석 섹션", list(NAVIGATION[group_name].keys()))
    module_name, function_name = NAVIGATION[group_name][section_name]

    start = perf_counter()
    with st.spinner(f"{section_name} 로딩 중..."):
        render = _resolve_renderer(module_name, function_name)
        render()
    elapsed = perf_counter() - start

    logger.info(f"dashboard render complete | group={group_name} | section={section_name} | elapsed={elapsed:.3f}s")
    st.sidebar.metric("마지막 렌더", f"{elapsed:.2f}초")


if __name__ == "__main__":
    main()