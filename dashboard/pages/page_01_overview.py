"""Page 01 – 종합 현황.

KPI 카드, 서울 전체 평균 매매가 추이, 자치구별 히트맵 등을 표시한다.
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

from config.settings import SEOUL_REGIONS
from dashboard.data_loader import load_trade_summary, load_macro_monthly


def render() -> None:
    """종합 현황 페이지를 렌더링한다."""
    st.header("종합 현황")

    trade_df = load_trade_summary()
    macro_df = load_macro_monthly()

    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다. 먼저 파이프라인을 실행해주세요.")
        return

    # --- KPI 카드 ---
    seoul_codes = set(SEOUL_REGIONS.keys())
    seoul_df = trade_df[trade_df["_lawd_cd"].isin(seoul_codes)].copy()

    if not seoul_df.empty:
        latest_ym = seoul_df["ym"].max()
        prev_ym_candidates = seoul_df[seoul_df["ym"] < latest_ym]["ym"]
        prev_ym = prev_ym_candidates.max() if not prev_ym_candidates.empty else None
        latest_ym_label = f"{latest_ym[:4]}-{latest_ym[4:]}" if isinstance(latest_ym, str) and len(latest_ym) == 6 else latest_ym

        latest = seoul_df[seoul_df["ym"] == latest_ym]
        avg_price = latest["평균거래금액"].mean()
        total_count = latest["거래건수"].sum()

        cols = st.columns(4)
        cols[0].metric(
            "서울 평균 매매가 (만원)",
            f"{avg_price:,.0f}" if avg_price == avg_price else "N/A",
        )
        cols[1].metric("서울 거래건수", f"{total_count:,}")
        cols[2].metric("기준 연월", latest_ym_label)

        if not macro_df.empty and "bok_rate" in macro_df.columns:
            latest_rate = macro_df["bok_rate"].dropna().iloc[-1] if not macro_df["bok_rate"].dropna().empty else None
            cols[3].metric("한국 기준금리", f"{latest_rate}%" if latest_rate is not None else "N/A")
        else:
            cols[3].metric("한국 기준금리", "N/A")

    # --- 서울 전체 월별 평균 매매가 추이 ---
    st.subheader("서울 전체 월별 평균 매매가 추이")

    if not seoul_df.empty:
        monthly_avg = (
            seoul_df.groupby(["ym", "date"])["평균거래금액"]
            .mean()
            .reset_index()
            .sort_values("date")
        )
        fig = px.line(
            monthly_avg,
            x="date",
            y="평균거래금액",
            title="서울 월별 평균 아파트 매매가 (만원)",
            labels={"date": "연월", "평균거래금액": "평균 매매가 (만원)"},
        )
        fig.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    # --- 자치구별 × 연도별 히트맵 ---
    st.subheader("자치구별 × 연도별 평균 거래금액 히트맵")

    if not seoul_df.empty:
        seoul_df["year"] = seoul_df["ym"].str[:4]
        heatmap_data = seoul_df.pivot_table(
            index="_region_name",
            columns="year",
            values="평균거래금액",
            aggfunc="mean",
        )
        fig_heat = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            color_continuous_scale="RdYlGn_r",
            labels={"color": "평균 매매가 (만원)"},
            title="자치구별 × 연도별 평균 거래금액",
            aspect="auto",
        )
        st.plotly_chart(fig_heat, width="stretch")

    # --- 거래건수 추이 ---
    st.subheader("서울 월별 총 거래건수")

    if not seoul_df.empty:
        monthly_count = (
            seoul_df.groupby(["ym", "date"])["거래건수"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        fig_count = px.bar(
            monthly_count,
            x="date",
            y="거래건수",
            title="서울 월별 아파트 매매 거래건수",
            labels={"date": "연월", "거래건수": "거래건수"},
        )
        fig_count.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
        st.plotly_chart(fig_count, width="stretch")
