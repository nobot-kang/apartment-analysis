"""Streamlit 대시보드 메인 앱.

사이드바 필터와 멀티페이지 레이아웃을 설정한다.
"""

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
    st.sidebar.markdown("서울·경기 아파트 실거래가와 거시경제 지표를 통합 분석합니다.")

    pages = {
        "종합 현황": "dashboard/pages/01_overview.py",
        "매매가 분석": "dashboard/pages/02_trade_price.py",
        "전월세 분석": "dashboard/pages/03_rent_price.py",
        "거시지표": "dashboard/pages/04_macro_indicators.py",
        "복합 상관관계": "dashboard/pages/05_correlation.py",
        "데이터 진단 & 시장 스냅샷": "dashboard/pages/06_market_snapshot.py",
    }

    selection = st.sidebar.radio("페이지 선택", list(pages.keys()))

    if selection == "종합 현황":
        from dashboard.pages.page_01_overview import render
        render()
    elif selection == "매매가 분석":
        from dashboard.pages.page_02_trade_price import render
        render()
    elif selection == "전월세 분석":
        from dashboard.pages.page_03_rent_price import render
        render()
    elif selection == "거시지표":
        from dashboard.pages.page_04_macro_indicators import render
        render()
    elif selection == "복합 상관관계":
        from dashboard.pages.page_05_correlation import render
        render()
    elif selection == "데이터 진단 & 시장 스냅샷":
        from dashboard.pages.page_06_market_snapshot import render
        render()


if __name__ == "__main__":
    main()
