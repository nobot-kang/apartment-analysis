"""Page 03 – 전월세 분석.

전세/월세 구분, 보증금 추이, 월세 분포 등을 표시한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dashboard.data_loader import load_rent_summary, load_trade_summary, get_region_options


def render() -> None:
    """전월세 분석 페이지를 렌더링한다."""
    st.header("전월세 분석")

    rent_df = load_rent_summary()
    if rent_df.empty:
        st.warning("전월세 집계 데이터가 없습니다.")
        return

    regions = get_region_options()

    # --- 사이드바 필터 ---
    selected_names = st.sidebar.multiselect(
        "지역 선택 (전월세)",
        options=list(regions.values()),
        default=["강남구", "서초구", "송파구"],
        key="rent_regions",
    )
    name_to_code = {v: k for k, v in regions.items()}
    selected_codes = [name_to_code[n] for n in selected_names if n in name_to_code]

    if not selected_codes:
        st.info("사이드바에서 지역을 선택해주세요.")
        return

    filtered = rent_df[rent_df["_lawd_cd"].isin(selected_codes)].copy()
    filtered = filtered.sort_values("ym")

    # --- 전세 / 월세 탭 ---
    tab_jeonse, tab_wolse = st.tabs(["전세", "월세"])

    has_rent_type = "rentType" in filtered.columns

    with tab_jeonse:
        st.subheader("전세 보증금 추이")
        if has_rent_type:
            jeonse = filtered[filtered["rentType"].str.strip() == "전세"]
        else:
            jeonse = filtered

        if not jeonse.empty and "평균보증금" in jeonse.columns:
            fig = px.line(
                jeonse,
                x="date",
                y="평균보증금",
                color="_region_name",
                title="전세 평균 보증금 추이 (만원)",
                labels={"date": "연월", "평균보증금": "평균 보증금 (만원)", "_region_name": "지역"},
            )
            fig.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
            st.plotly_chart(fig, width="stretch")

            # 전세가율 (매매가 대비)
            trade_df = load_trade_summary()
            if not trade_df.empty:
                st.subheader("전세가율 (전세보증금 / 매매가)")
                trade_avg = (
                    trade_df.groupby(["ym", "date", "_lawd_cd"])["평균거래금액"]
                    .mean()
                    .reset_index()
                )
                jeonse_avg = (
                    jeonse.groupby(["ym", "date", "_lawd_cd"])["평균보증금"]
                    .mean()
                    .reset_index()
                )

                merged = jeonse_avg.merge(trade_avg, on=["ym", "date", "_lawd_cd"], how="inner")
                merged["전세가율"] = (merged["평균보증금"] / merged["평균거래금액"]) * 100
                merged = merged.merge(
                    rent_df[["_lawd_cd", "_region_name"]].drop_duplicates(),
                    on="_lawd_cd",
                    how="left",
                )

                fig_ratio = px.line(
                    merged,
                    x="date",
                    y="전세가율",
                    color="_region_name",
                    title="전세가율 추이 (%)",
                    labels={"date": "연월", "전세가율": "전세가율 (%)", "_region_name": "지역"},
                )
                fig_ratio.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
                st.plotly_chart(fig_ratio, width="stretch")
        else:
            st.info("전세 데이터가 없습니다.")

    with tab_wolse:
        st.subheader("월세 추이")
        if has_rent_type:
            wolse = filtered[filtered["rentType"].str.strip() == "월세"]
        else:
            wolse = pd.DataFrame()

        if not wolse.empty and "평균월세" in wolse.columns:
            fig_wolse = px.line(
                wolse,
                x="date",
                y="평균월세",
                color="_region_name",
                title="월세 평균 월세금액 추이 (만원)",
                labels={"date": "연월", "평균월세": "평균 월세 (만원)", "_region_name": "지역"},
            )
            fig_wolse.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
            st.plotly_chart(fig_wolse, width="stretch")

            # 보증금 × 월세 산점도
            if "평균보증금" in wolse.columns:
                st.subheader("보증금 × 월세 분포")
                fig_scatter = px.scatter(
                    wolse,
                    x="평균보증금",
                    y="평균월세",
                    color="_region_name",
                    title="보증금 vs 월세",
                    labels={"평균보증금": "평균 보증금 (만원)", "평균월세": "평균 월세 (만원)", "_region_name": "지역"},
                )
                st.plotly_chart(fig_scatter, width="stretch")
        else:
            st.info("월세 데이터가 없습니다.")

    # --- 거래건수 비교 ---
    st.subheader("전월세 거래건수 추이")
    if has_rent_type:
        count_by_type = (
            filtered.groupby(["ym", "date", "rentType"])["거래건수"]
            .sum()
            .reset_index()
        )
        fig_type = px.bar(
            count_by_type,
            x="date",
            y="거래건수",
            color="rentType",
            barmode="group",
            text="거래건수",
            title="전세/월세 거래건수 추이",
            labels={"date": "연월", "거래건수": "거래건수", "rentType": "유형"},
        )
        fig_type.update_traces(textposition="outside")
        fig_type.update_xaxes(tickformat="%Y-%m", dtick="M3", tickangle=-45)
        st.plotly_chart(fig_type, width="stretch")
