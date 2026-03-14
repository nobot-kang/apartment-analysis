"""Page 03 - 전월세 심화 분석."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import aggregate_rent_scope, get_scope_codes
from analysis.level1 import build_jeonse_ratio_chart, load_jeonse_ratio
from analysis.level2 import build_conversion_rate_chart, load_conversion_rate_data
from dashboard.data_loader import load_rent_summary


@st.cache_data(ttl=3600)
def _get_rent_scope(scope_name: str) -> pd.DataFrame:
    return aggregate_rent_scope(load_rent_summary(), get_scope_codes(scope_name), scope_name)


@st.cache_data(ttl=3600)
def _get_ratio() -> pd.DataFrame:
    return load_jeonse_ratio()


@st.cache_data(ttl=3600)
def _get_conversion(scope_name: str) -> pd.DataFrame:
    return load_conversion_rate_data(get_scope_codes(scope_name), scope_name)


def render() -> None:
    st.header("전월세 분석")

    rent_df = load_rent_summary()
    if rent_df.empty:
        st.warning("전월세 집계 데이터가 없습니다.")
        return

    scope_name = st.selectbox("분석 범위", ["서울 전체", "경기 전체", "수도권 전체", *sorted(rent_df["_region_name"].unique())], index=0)
    scope_df = _get_rent_scope(scope_name)
    ratio_df = _get_ratio()
    conversion_df = _get_conversion(scope_name)

    tab1, tab2, tab3 = st.tabs(["전세/월세 추이", "전세가율", "전월세 전환율"])

    with tab1:
        if scope_df.empty:
            st.info("선택 범위의 전월세 데이터가 없습니다.")
        else:
            fig_deposit = px.line(
                scope_df,
                x="date",
                y="평균보증금",
                color="rentType",
                title=f"{scope_name} 평균 보증금 추이",
                labels={"date": "월", "평균보증금": "평균 보증금 (만원)", "rentType": "유형"},
            )
            st.plotly_chart(fig_deposit, width="stretch")

            fig_count = px.bar(
                scope_df,
                x="date",
                y="거래건수",
                color="rentType",
                barmode="group",
                title=f"{scope_name} 전세/월세 거래건수",
                labels={"date": "월", "거래건수": "거래건수", "rentType": "유형"},
            )
            st.plotly_chart(fig_count, width="stretch")

            wolse = scope_df[scope_df["rentType"] == "월세"]
            if not wolse.empty:
                fig_scatter = px.scatter(
                    wolse,
                    x="평균보증금",
                    y="평균월세",
                    size="거래건수",
                    title=f"{scope_name} 보증금-월세 분포",
                    labels={"평균보증금": "평균 보증금 (만원)", "평균월세": "평균 월세 (만원)"},
                )
                st.plotly_chart(fig_scatter, width="stretch")

    with tab2:
        available_regions = sorted(ratio_df["_region_name"].unique()) if not ratio_df.empty else []
        default_index = available_regions.index(scope_name) if scope_name in available_regions else 0
        region_name = st.selectbox("전세가율 지역", available_regions, index=default_index) if available_regions else None
        if region_name:
            st.plotly_chart(build_jeonse_ratio_chart(ratio_df, region_name), width="stretch")

    with tab3:
        st.plotly_chart(build_conversion_rate_chart(conversion_df, scope_name), width="stretch")
        if not conversion_df.empty:
            latest = conversion_df.sort_values("date").iloc[-1]
            st.metric("최근 전환율", f"{latest['conversion_rate']:.2f}%", help="월세를 연환산하여 전세보증금 차이로 나눈 값입니다.")

