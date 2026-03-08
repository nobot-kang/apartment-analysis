"""Page 02 – 매매가 분석.

지역 × 면적대별 매매가 시계열, 거래량 바차트, YoY 상승률 등을 표시한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dashboard.data_loader import load_trade_summary, get_region_options
from analysis.trend import add_trend_columns


def render() -> None:
    """매매가 분석 페이지를 렌더링한다."""
    st.header("매매가 분석")

    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    regions = get_region_options()

    # --- 사이드바 필터 ---
    selected_names = st.sidebar.multiselect(
        "지역 선택",
        options=list(regions.values()),
        default=["강남구", "서초구", "송파구"],
    )
    name_to_code = {v: k for k, v in regions.items()}
    selected_codes = [name_to_code[n] for n in selected_names if n in name_to_code]

    if not selected_codes:
        st.info("사이드바에서 지역을 선택해주세요.")
        return

    filtered = trade_df[trade_df["_lawd_cd"].isin(selected_codes)].copy()
    filtered = filtered.sort_values("ym")

    # --- 지역별 매매가 시계열 ---
    st.subheader("지역별 평균 매매가 추이")
    fig = px.line(
        filtered,
        x="date",
        y="평균거래금액",
        color="_region_name",
        title="지역별 월 평균 매매가 (만원)",
        labels={"date": "연월", "평균거래금액": "평균 매매가 (만원)", "_region_name": "지역"},
    )
    fig.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    # --- 거래량 바차트 ---
    st.subheader("월별 거래건수")
    fig_count = px.bar(
        filtered,
        x="date",
        y="거래건수",
        color="_region_name",
        barmode="group",
        title="월별 아파트 매매 거래건수",
        labels={"date": "연월", "거래건수": "거래건수", "_region_name": "지역"},
    )
    fig_count.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
    st.plotly_chart(fig_count, width="stretch")

    # --- 면적대별 가격 (있는 경우) ---
    area_cols = [c for c in filtered.columns if c.startswith("평균거래금액_")]
    if area_cols:
        st.subheader("면적대별 평균 매매가")

        selected_region = st.selectbox("지역 선택 (면적대별)", selected_names)
        region_code = name_to_code.get(selected_region)
        region_data = filtered[filtered["_lawd_cd"] == region_code].copy()

        if not region_data.empty:
            area_melted = region_data.melt(
                id_vars=["ym", "date"],
                value_vars=area_cols,
                var_name="면적대",
                value_name="면적대별평균거래금액",
            )
            area_melted["면적대"] = area_melted["면적대"].str.replace("평균거래금액_", "")

            fig_area = px.line(
                area_melted,
                x="date",
                y="면적대별평균거래금액",
                color="면적대",
                title=f"{selected_region} 면적대별 평균 매매가",
                labels={"date": "연월", "면적대별평균거래금액": "평균 매매가 (만원)"},
            )
            fig_area.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
            st.plotly_chart(fig_area, width="stretch")

    # --- YoY 상승률 ---
    st.subheader("전년동월비(YoY) 변화율")

    for name in selected_names:
        code = name_to_code.get(name)
        region_data = filtered[filtered["_lawd_cd"] == code].copy().sort_values("ym").reset_index(drop=True)

        if len(region_data) > 12:
            with_trend = add_trend_columns(region_data, "평균거래금액")
            fig_yoy = px.line(
                with_trend,
                x="date",
                y="평균거래금액_YoY",
                title=f"{name} 매매가 YoY 변화율 (%)",
                labels={"date": "연월", "평균거래금액_YoY": "YoY (%)"},
            )
            fig_yoy.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_yoy.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
            st.plotly_chart(fig_yoy, width="stretch")
