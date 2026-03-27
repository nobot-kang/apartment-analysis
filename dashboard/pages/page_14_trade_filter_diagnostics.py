"""거래 필터 진단 페이지.

raw 매매 데이터 기준으로 취소거래/직거래 비율을 연도·지역별로 점검한다.
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

from config.settings import GYEONGGI_REGIONS, SEOUL_REGIONS
from dashboard.data_loader import load_trade_filter_yearly_summary


def _region_options() -> dict[str, str]:
    options = {"ALL": "전체", "SEOUL": "서울 전체", "GYEONGGI": "경기 전체"}
    options.update(SEOUL_REGIONS)
    options.update(GYEONGGI_REGIONS)
    return options


def _build_ratio_chart(df: pd.DataFrame, value_col: str, title: str, y_label: str):
    if df.empty:
        return px.line(title=title)

    fig = px.line(
        df.sort_values(["region_name", "year"]),
        x="year",
        y=value_col,
        color="region_name",
        markers=True,
        labels={"year": "연도", value_col: y_label, "region_name": "지역"},
        title=title,
    )
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    return fig


def render_trade_filter_diagnostics() -> None:
    st.header("거래 필터 진단")
    st.markdown(
        "매매 raw 데이터에서 **취소거래**와 **직거래**가 지역별로 얼마나 발생했는지 확인합니다. "
        "분석용 `apt_trade_*` 전처리에서는 취소거래와 직거래를 제외합니다."
    )

    df = load_trade_filter_yearly_summary()
    if df.empty:
        st.warning("거래 필터 요약 데이터가 없습니다. 먼저 전처리를 다시 실행해주세요.")
        st.code("uv run python pipelines/data_preprocessing.py", language="bash")
        return

    region_opts = _region_options()
    available_codes = [code for code in region_opts if code in set(df["sggCd"].astype(str))]
    default_codes = [code for code in ["ALL", "SEOUL", "GYEONGGI"] if code in available_codes]
    selected_codes = st.multiselect(
        "지역 선택",
        options=available_codes,
        default=default_codes,
        format_func=lambda code: region_opts.get(code, code),
    )
    if not selected_codes:
        st.info("최소 1개 지역을 선택해주세요.")
        return

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    selected_years = st.slider("연도 범위", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    filtered = df[
        df["sggCd"].astype(str).isin(selected_codes)
        & df["year"].between(selected_years[0], selected_years[1])
    ].copy()
    filtered["region_name"] = pd.Categorical(
        filtered["region_name"],
        categories=[region_opts[code] for code in selected_codes if code in region_opts],
        ordered=True,
    )
    filtered = filtered.sort_values(["region_name", "year"])

    col1, col2 = st.columns(2)
    latest_year = filtered["year"].max()
    latest = filtered[filtered["year"] == latest_year]
    col1.metric("최근 연도", f"{latest_year}")
    col2.metric("선택 지역 수", f"{latest['sggCd'].nunique():,}")

    st.plotly_chart(
        _build_ratio_chart(
            filtered,
            "cancel_ratio_pct",
            "지역별 연도별 전체 거래 대비 취소거래 비율",
            "취소거래 비율",
        ),
        width="stretch",
    )
    st.plotly_chart(
        _build_ratio_chart(
            filtered,
            "direct_ratio_pct",
            "지역별 연도별 전체 거래 대비 직거래 비율",
            "직거래 비율",
        ),
        width="stretch",
    )

    st.markdown("#### 연도별 요약 표")
    display_cols = [
        "year",
        "region_name",
        "total_trade_count",
        "cancel_trade_count",
        "cancel_ratio_pct",
        "direct_trade_count",
        "direct_ratio_pct",
    ]
    table = filtered[display_cols].copy()
    table["cancel_ratio_pct"] = table["cancel_ratio_pct"].round(2)
    table["direct_ratio_pct"] = table["direct_ratio_pct"].round(2)
    st.dataframe(table, width="stretch", height=420)
