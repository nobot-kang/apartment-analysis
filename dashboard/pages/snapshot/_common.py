"""snapshot 공통 헬퍼 — streamlit 의존 없음."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import SEOUL_REGIONS, GYEONGGI_REGIONS


def _region_options() -> dict[str, str]:
    """지역 선택 옵션 딕셔너리를 반환한다."""
    options = {"ALL": "전체", "SEOUL": "서울 전체", "GYEONGGI": "경기 전체"}
    options.update(SEOUL_REGIONS)
    options.update(GYEONGGI_REGIONS)
    return options


def _plotly_line(df: pd.DataFrame, x: str, y_cols: list[str], title: str,
                 y_label: str = "", colors: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    palette = colors or px.colors.qualitative.Plotly
    for i, col in enumerate(y_cols):
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col],
            name=col,
            mode="lines",
            line=dict(color=palette[i % len(palette)], width=2),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="연월",
        yaxis_title=y_label,
        legend=dict(orientation="h", y=-0.2),
        height=380,
    )
    fig.update_xaxes(tickformat="%Y-%m", dtick="M6", tickangle=-30)
    return fig
