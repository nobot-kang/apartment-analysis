"""Page 04 – 거시지표.

기준금리, CPI, M2, 금가격, 유가, 환율 등 거시경제 지표를 시각화한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dashboard.data_loader import load_macro_monthly, load_trade_summary
from config.settings import SEOUL_REGIONS


def render() -> None:
    """거시지표 페이지를 렌더링한다."""
    st.header("거시지표")

    macro_df = load_macro_monthly()
    if macro_df.empty:
        st.warning("거시지표 데이터가 없습니다.")
        return

    macro_df = macro_df.sort_values("date")

    # --- 이중 Y축: 매매가 vs 기준금리 ---
    st.subheader("서울 평균 매매가 vs 기준금리")

    trade_df = load_trade_summary()
    if not trade_df.empty:
        seoul_codes = set(SEOUL_REGIONS.keys())
        seoul_monthly = (
            trade_df[trade_df["_lawd_cd"].isin(seoul_codes)]
            .groupby("ym")["평균거래금액"]
            .mean()
            .reset_index()
            .sort_values("ym")
        )
        seoul_monthly["date"] = pd.to_datetime(seoul_monthly["ym"], format="%Y%m")

        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(
            go.Scatter(
                x=seoul_monthly["date"],
                y=seoul_monthly["평균거래금액"],
                name="서울 평균 매매가 (만원)",
                line={"color": "royalblue"},
            ),
            secondary_y=False,
        )

        if "bok_rate" in macro_df.columns:
            fig_dual.add_trace(
                go.Scatter(
                    x=macro_df["date"],
                    y=macro_df["bok_rate"],
                    name="한국 기준금리 (%)",
                    line={"color": "red", "dash": "dot"},
                ),
                secondary_y=True,
            )
        if "fed_rate" in macro_df.columns:
            fig_dual.add_trace(
                go.Scatter(
                    x=macro_df["date"],
                    y=macro_df["fed_rate"],
                    name="미국 기준금리 (%)",
                    line={"color": "orange", "dash": "dot"},
                ),
                secondary_y=True,
            )

        fig_dual.update_layout(title="서울 아파트 매매가 vs 기준금리")
        fig_dual.update_yaxes(title_text="평균 매매가 (만원)", secondary_y=False)
        fig_dual.update_yaxes(title_text="기준금리 (%)", secondary_y=True)
        st.plotly_chart(fig_dual, width="stretch")

    # --- 서브플롯: 각 지표 시계열 ---
    st.subheader("거시경제 지표 시계열")

    indicator_labels = {
        "bok_rate": "한국 기준금리 (%)",
        "fed_rate": "미국 기준금리 (%)",
        "cpi_kr": "한국 CPI",
        "cpi_us": "미국 CPI",
        "m2": "M2 (십억원)",
        "gold": "금 가격 (USD/oz)",
        "oil": "유가 WTI (USD)",
        "usdkrw": "원달러 환율",
    }

    available = [col for col in indicator_labels if col in macro_df.columns]

    if available:
        n_cols = 3
        n_rows = (len(available) + n_cols - 1) // n_cols

        fig_sub = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[indicator_labels[c] for c in available],
        )

        for idx, col in enumerate(available):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1
            fig_sub.add_trace(
                go.Scatter(
                    x=macro_df["date"],
                    y=macro_df[col],
                    name=indicator_labels[col],
                    showlegend=False,
                ),
                row=row,
                col=col_pos,
            )

        fig_sub.update_layout(
            height=300 * n_rows,
            title_text="거시경제 지표 서브플롯",
        )
        st.plotly_chart(fig_sub, width="stretch")

    # --- 지표별 증감률 테이블 ---
    st.subheader("지표별 전월비/전년비 증감률")

    if available:
        summary_rows = []
        for col in available:
            series = macro_df[col].dropna()
            if len(series) < 2:
                continue
            latest = series.iloc[-1]
            mom = ((series.iloc[-1] / series.iloc[-2]) - 1) * 100 if series.iloc[-2] != 0 else None
            yoy = None
            if len(series) >= 13:
                yoy = ((series.iloc[-1] / series.iloc[-13]) - 1) * 100 if series.iloc[-13] != 0 else None

            summary_rows.append({
                "지표": indicator_labels[col],
                "최신값": f"{latest:,.2f}",
                "전월비 (%)": f"{mom:+.2f}" if mom is not None else "N/A",
                "전년비 (%)": f"{yoy:+.2f}" if yoy is not None else "N/A",
            })

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")
