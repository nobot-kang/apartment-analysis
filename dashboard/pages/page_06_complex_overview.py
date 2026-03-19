"""Dashboard page for complex-level descriptive analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.complex_analysis import (
    build_complex_profile_chart,
    build_complex_profile_frame,
    build_coverage_report,
    build_density_heatmap,
    build_density_matrix,
    build_land_premium_chart,
    build_latest_snapshot,
    build_parking_premium_chart,
    build_parking_premium_frame,
    build_scale_premium_chart,
    build_scale_premium_frame,
)
from dashboard.data_loader import load_complex_master, load_complex_monthly_panel


@st.cache_data(ttl=3600, show_spinner=False)
def _get_master_diagnostics():
    master_df = load_complex_master()
    return build_complex_profile_frame(master_df), build_coverage_report(master_df)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_snapshot(months: int):
    return build_latest_snapshot(load_complex_monthly_panel(), months=months)


def _parking_coverage_pct() -> float:
    coverage_df = build_coverage_report(load_complex_master())
    row = coverage_df[coverage_df["feature"] == "세대당 주차대수"]
    if row.empty:
        return 0.0
    return float(row["coverage_pct"].iloc[0])


def render_complex_profile() -> None:
    st.header("Complex Level 1 - 단지 특성 프로파일링")
    master_df = load_complex_master()
    if master_df.empty:
        st.warning("complex_master.parquet 가 없습니다. 집계 파이프라인을 먼저 실행해 주세요.")
        return

    profile_df, coverage_df = _get_master_diagnostics()
    cols = st.columns(4)
    cols[0].metric("단지 수", f"{master_df['aptSeq'].nunique():,}")
    cols[1].metric("법정동 수", f"{master_df['dong_repr'].nunique():,}" if "dong_repr" in master_df.columns else "N/A")
    cols[2].metric("중위 세대수", f"{master_df['household_count'].median():,.0f}" if "household_count" in master_df.columns else "N/A")
    parking_cov = coverage_df.loc[coverage_df["feature"] == "세대당 주차대수", "coverage_pct"]
    cols[3].metric("주차 정보 커버리지", f"{parking_cov.iloc[0]:.1f}%" if not parking_cov.empty else "0.0%")
    st.plotly_chart(build_complex_profile_chart(profile_df), width="stretch")
    st.dataframe(coverage_df, width="stretch", hide_index=True)


def render_complex_scale_premium() -> None:
    st.header("Complex Level 1 - 대단지 프리미엄")
    months = int(st.select_slider("집계 기간", options=[6, 12, 24], value=12, key="complex_scale_months"))
    snapshot_df = _get_snapshot(months)
    if snapshot_df.empty:
        st.warning("최근 단지 스냅샷 데이터가 없습니다.")
        return

    scale_df = build_scale_premium_frame(snapshot_df)
    st.plotly_chart(build_scale_premium_chart(scale_df), width="stretch")
    st.dataframe(scale_df, width="stretch", hide_index=True)


def render_complex_parking_premium() -> None:
    st.header("Complex Level 1 - 주차 프리미엄")
    coverage_pct = _parking_coverage_pct()
    if coverage_pct < 5:
        st.warning("현재 원천 데이터에 총주차수 컬럼이 거의 없어 주차 분석은 제한적으로만 표시됩니다.")

    months = int(st.select_slider("집계 기간", options=[6, 12, 24], value=12, key="complex_parking_months"))
    snapshot_df = _get_snapshot(months)
    parking_df = build_parking_premium_frame(snapshot_df)
    if parking_df.empty:
        st.info("주차 정보가 충분한 단지가 아직 없습니다.")
        return

    st.plotly_chart(build_parking_premium_chart(parking_df), width="stretch")
    st.dataframe(parking_df, width="stretch", hide_index=True)


def render_complex_density_premium() -> None:
    st.header("Complex Level 1 - 용적률/건폐율 밀도 프리미엄")
    months = int(st.select_slider("집계 기간", options=[6, 12, 24], value=12, key="complex_density_months"))
    snapshot_df = _get_snapshot(months)
    metric = st.radio(
        "가격 지표",
        options=["trade_price_per_m2", "jeonse_deposit_per_m2", "wolse_monthly_rent_per_m2"],
        format_func=lambda value: {
            "trade_price_per_m2": "매매 가격",
            "jeonse_deposit_per_m2": "전세 가격",
            "wolse_monthly_rent_per_m2": "월세 가격",
        }[value],
        horizontal=True,
        key="complex_density_metric",
    )
    matrix_df = build_density_matrix(snapshot_df, value_col=metric)
    st.plotly_chart(build_density_heatmap(matrix_df, "밀도 조합별 가격 매트릭스"), width="stretch")
    if not matrix_df.empty:
        st.dataframe(matrix_df, width="stretch")


def render_complex_land_premium() -> None:
    st.header("Complex Level 1 - 세대당 평균대지면적 프리미엄")
    months = int(st.select_slider("집계 기간", options=[6, 12, 24], value=12, key="complex_land_months"))
    snapshot_df = _get_snapshot(months)
    if snapshot_df.empty:
        st.warning("최근 단지 스냅샷 데이터가 없습니다.")
        return

    st.plotly_chart(build_land_premium_chart(snapshot_df), width="stretch")
    valid = snapshot_df[["avg_land_area_per_household", "trade_price_per_m2"]].dropna()
    if not valid.empty:
        st.metric("상관계수", f"{valid['avg_land_area_per_household'].corr(valid['trade_price_per_m2']):.3f}")
