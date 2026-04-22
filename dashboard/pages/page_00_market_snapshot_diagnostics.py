"""Page 06 – A. 데이터 진단 & 시장 스냅샷.

A-1 월별 거래량·중위 ㎡당 가격·분산 추이
A-2 면적 믹스 변화 & 구성효과 분해
A-3 이상치·오류·비정상 거래 탐지
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dashboard.data_loader import (
    load_snapshot_monthly_trade,
    load_snapshot_monthly_rent,
    load_snapshot_area_mix,
    load_snapshot_complex_market_price,
    load_snapshot_outliers,
)
from dashboard.pages.snapshot._common import _region_options
from dashboard.pages.snapshot.tab_a1_monthly_trend import _render_a1
from dashboard.pages.snapshot.tab_a2_area_mix import _render_a2
from dashboard.pages.snapshot.tab_a3_outliers import _render_a3
# TODO(cleanup): remove once no repo-wide usage of _resolve_color_col via this module — check
# (a) direct import "from dashboard.pages.page_00_market_snapshot_diagnostics import _resolve_color_col" (별칭 포함),
# (b) attribute access "page_00_market_snapshot_diagnostics._resolve_color_col" 양쪽 모두 0건인지.
from dashboard.pages.snapshot.a3_filters import _resolve_color_col  # noqa: F401 — backward-compat re-export

__all__ = ["render_snapshot", "_resolve_color_col"]


def render_snapshot() -> None:
    """시장 스냅샷 페이지를 렌더링한다."""
    st.header("A. 데이터 진단 & 시장 스냅샷")
    st.markdown(
        "아파트 매매·전월세 실거래 데이터의 기초 현황을 진단하고, "
        "시장의 흐름을 한눈에 파악합니다."
    )

    # 데이터 로드
    trade_df = load_snapshot_monthly_trade()
    rent_df = load_snapshot_monthly_rent()
    area_mix_df = load_snapshot_area_mix()
    outliers_df = load_snapshot_outliers()
    market_price_df = load_snapshot_complex_market_price()

    pipeline_not_run = trade_df.empty and area_mix_df.empty

    if pipeline_not_run:
        st.warning(
            "집계 데이터가 없습니다. 먼저 파이프라인을 실행해주세요.\n\n"
            "```bash\npython pipelines/market_snapshot_pipeline.py\n```"
        )
        return

    # 사이드바 지역 선택
    region_opts = _region_options()
    region_display = list(region_opts.values())
    region_codes = list(region_opts.keys())

    selected_name = st.sidebar.selectbox(
        "지역 선택 (Section A)",
        region_display,
        index=0,
        key="snapshot_region",
    )
    selected_code = region_codes[region_display.index(selected_name)]

    # 기간 범위 슬라이더
    if not trade_df.empty and "month" in trade_df.columns:
        min_date = trade_df["month"].min()
        max_date = trade_df["month"].max()
        min_year = int(min_date.year)
        max_year = int(max_date.year)

        year_range = st.sidebar.slider(
            "조회 연도 범위",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="snapshot_year_range",
        )
        # 기간 필터 적용
        trade_df = trade_df[
            (trade_df["month"].dt.year >= year_range[0])
            & (trade_df["month"].dt.year <= year_range[1])
        ]
        if not rent_df.empty:
            rent_df = rent_df[
                (rent_df["month"].dt.year >= year_range[0])
                & (rent_df["month"].dt.year <= year_range[1])
            ]
        if not area_mix_df.empty:
            area_mix_df = area_mix_df[
                (area_mix_df["month"].dt.year >= year_range[0])
                & (area_mix_df["month"].dt.year <= year_range[1])
            ]

    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "A-1. 월별 거래량·가격",
        "A-2. 면적 믹스",
        "A-3. 이상치 탐지",
    ])

    with tab1:
        _render_a1(trade_df, rent_df, selected_code)

    with tab2:
        _render_a2(area_mix_df, selected_code)

    with tab3:
        _render_a3(outliers_df, selected_code, market_price_df)
