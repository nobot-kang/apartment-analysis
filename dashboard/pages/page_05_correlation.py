"""Page 05 – 복합 상관관계.

상관계수 히트맵, 시차 상관분석, 산점도 매트릭스 등을 표시한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import SEOUL_REGIONS
from dashboard.data_loader import load_macro_monthly, load_trade_summary
from analysis.correlation import correlation_matrix, lagged_correlation, simple_regression


def _build_combined_df() -> pd.DataFrame:
    """매매가 + 거시지표를 결합한 분석용 DataFrame을 생성한다.

    Returns:
        월별 통합 DataFrame.
    """
    trade_df = load_trade_summary()
    macro_df = load_macro_monthly()

    if trade_df.empty or macro_df.empty:
        return pd.DataFrame()

    seoul_codes = set(SEOUL_REGIONS.keys())
    seoul_monthly = (
        trade_df[trade_df["_lawd_cd"].isin(seoul_codes)]
        .groupby("ym")
        .agg(평균매매가=("평균거래금액", "mean"))
        .reset_index()
        .sort_values("ym")
    )
    seoul_monthly["date"] = pd.to_datetime(seoul_monthly["ym"], format="%Y%m")

    macro_df = macro_df.sort_values("date")

    combined = seoul_monthly.merge(macro_df, on="date", how="inner")
    return combined


def render() -> None:
    """복합 상관관계 페이지를 렌더링한다."""
    st.header("복합 상관관계 분석")

    combined = _build_combined_df()

    if combined.empty:
        st.warning("데이터가 부족합니다. 집계 파이프라인을 먼저 실행해주세요.")
        return

    # 분석 대상 컬럼
    analysis_cols = ["평균매매가"]
    indicator_cols = ["bok_rate", "fed_rate", "cpi_kr", "cpi_us", "m2", "gold", "oil", "usdkrw"]
    available_indicators = [c for c in indicator_cols if c in combined.columns]
    all_cols = analysis_cols + available_indicators

    col_labels = {
        "평균매매가": "서울 평균매매가",
        "bok_rate": "한국 기준금리",
        "fed_rate": "미국 기준금리",
        "cpi_kr": "한국 CPI",
        "cpi_us": "미국 CPI",
        "m2": "M2",
        "gold": "금 가격",
        "oil": "유가",
        "usdkrw": "환율",
    }

    # --- 상관계수 히트맵 ---
    st.subheader("상관계수 히트맵")

    corr = correlation_matrix(combined, columns=all_cols)
    display_labels = [col_labels.get(c, c) for c in corr.columns]

    fig_corr = px.imshow(
        corr.values,
        x=display_labels,
        y=display_labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="상관계수 히트맵",
        text_auto=".2f",
        aspect="auto",
    )
    st.plotly_chart(fig_corr, width="stretch")

    # --- 시차 상관분석 ---
    st.subheader("시차(lag) 상관분석")

    lag_target = st.selectbox(
        "비교할 지표",
        options=available_indicators,
        format_func=lambda x: col_labels.get(x, x),
    )
    max_lag = st.slider("최대 시차 (개월)", min_value=1, max_value=24, value=12)

    if lag_target:
        lag_result = lagged_correlation(combined, "평균매매가", lag_target, max_lag=max_lag)

        fig_lag = px.bar(
            lag_result,
            x="lag",
            y="correlation",
            title=f"서울 평균매매가 vs {col_labels.get(lag_target, lag_target)} – 시차 상관계수",
            labels={"lag": "시차 (개월)", "correlation": "상관계수"},
            color="correlation",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
        )
        fig_lag.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_lag, width="stretch")

        peak_idx = lag_result["correlation"].abs().idxmax()
        peak = lag_result.iloc[peak_idx]
        st.info(
            f"최대 상관: lag={int(peak['lag'])}개월, "
            f"상관계수={peak['correlation']:.3f}"
        )

    # --- 산점도 + 회귀선 ---
    st.subheader("주요 지표 간 산점도")

    scatter_x = st.selectbox(
        "X축 지표",
        options=available_indicators,
        format_func=lambda x: col_labels.get(x, x),
        key="scatter_x",
    )
    scatter_y_options = ["평균매매가"] + [c for c in available_indicators if c != scatter_x]
    scatter_y = st.selectbox(
        "Y축 지표",
        options=scatter_y_options,
        format_func=lambda x: col_labels.get(x, x),
        key="scatter_y",
    )

    if scatter_x and scatter_y:
        reg = simple_regression(combined, scatter_x, scatter_y)
        subset = combined[[scatter_x, scatter_y]].dropna()

        fig_scatter = px.scatter(
            subset,
            x=scatter_x,
            y=scatter_y,
            title=f"{col_labels.get(scatter_x, scatter_x)} vs {col_labels.get(scatter_y, scatter_y)}",
            labels={
                scatter_x: col_labels.get(scatter_x, scatter_x),
                scatter_y: col_labels.get(scatter_y, scatter_y),
            },
            trendline="ols" if len(subset) >= 3 else None,
        )
        st.plotly_chart(fig_scatter, width="stretch")

        if not np.isnan(reg["r_squared"]):
            st.metric("R²", f"{reg['r_squared']:.4f}")
