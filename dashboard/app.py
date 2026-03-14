"""Streamlit 대시보드 메인 앱."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main() -> None:
    """Streamlit 앱 진입점."""
    st.set_page_config(
        page_title="부동산 실거래가 분석 대시보드",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("부동산 실거래가 분석")
    st.sidebar.markdown("기초 현황부터 거시지표, 고급 모델링까지 단계별 분석을 제공합니다.")

    pages = {
        "Level 1 · 기초 현황": "overview",
        "Level 2 · 심화 비교": "trade",
        "전월세 분석": "rent",
        "Level 3 · 거시 연계": "macro",
        "Level 4 · 고급 분석": "advanced",
    }
    selection = st.sidebar.radio("페이지 선택", list(pages.keys()))

    if selection == "Level 1 · 기초 현황":
        from dashboard.pages.page_01_overview import render
        render()
    elif selection == "Level 2 · 심화 비교":
        from dashboard.pages.page_02_trade_price import render
        render()
    elif selection == "전월세 분석":
        from dashboard.pages.page_03_rent_price import render
        render()
    elif selection == "Level 3 · 거시 연계":
        from dashboard.pages.page_04_macro_indicators import render
        render()
    elif selection == "Level 4 · 고급 분석":
        from dashboard.pages.page_05_correlation import render
        render()


if __name__ == "__main__":
    main()
