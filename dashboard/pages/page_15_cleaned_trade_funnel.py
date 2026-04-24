"""Page 15: 정제 단계별 거래 잔존율 — cleaned_apt_trade funnel."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipelines.cleaned_trade_pipeline import get_cleaned_trade_manifest


_PREPROCESSED_PLUS_DIR = _project_root / "data" / "preprocessed_plus"
_TRADE_FILTER_SUMMARY = _PREPROCESSED_PLUS_DIR / "trade_filter_yearly_summary.parquet"
_CLEANED_YEARLY_SUMMARY = _PREPROCESSED_PLUS_DIR / "cleaned_trade_yearly_summary.parquet"
_REASON_SUMMARY = _PREPROCESSED_PLUS_DIR / "cleaned_trade_outlier_reason_summary.parquet"


def _load_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """3-way join 용 summary parquet 3종을 로드한다. 실패 시 None."""
    for p in [_TRADE_FILTER_SUMMARY, _CLEANED_YEARLY_SUMMARY, _REASON_SUMMARY]:
        if not p.exists():
            return None
    try:
        tf = pd.read_parquet(_TRADE_FILTER_SUMMARY)
        cy = pd.read_parquet(_CLEANED_YEARLY_SUMMARY)
        rs = pd.read_parquet(_REASON_SUMMARY)
        return tf, cy, rs
    except Exception:
        return None


def render_funnel() -> None:
    """정제 단계별 거래 잔존율 대시보드를 렌더링한다."""
    st.header("정제 단계별 거래 잔존율")

    # -----------------------------------------------------------------------
    # Empty-state gate: manifest validity check
    # -----------------------------------------------------------------------
    try:
        get_cleaned_trade_manifest(verify_files=True)
    except (FileNotFoundError, RuntimeError) as exc:
        st.info(
            "정제 매매 데이터셋이 아직 검증되지 않았습니다. "
            "`uv run python scripts/build_cleaned_trade.py` 실행 후 다시 확인해 주세요. "
            f"(상세: {exc})"
        )
        return

    summaries = _load_summaries()
    if summaries is None:
        st.info(
            "요약 파일이 없습니다. "
            "`uv run python scripts/build_cleaned_trade.py` 실행 후 다시 확인해 주세요."
        )
        return

    tf_df, cy_df, rs_df = summaries

    # -----------------------------------------------------------------------
    # Region scope selector
    # -----------------------------------------------------------------------
    region_options = ["전체", "서울", "경기"]
    region = st.selectbox("지역", region_options, index=0)

    cy_region = cy_df[cy_df["region_scope"] == region].copy()
    tf_region = tf_df[
        tf_df["sggCd"].isin(
            {"ALL": ["ALL"], "전체": ["ALL"], "서울": ["SEOUL"], "경기": ["GYEONGGI"]}.get(
                region, ["ALL"]
            )
        )
    ].copy()

    if cy_region.empty:
        st.warning("선택한 지역의 데이터가 없습니다.")
        return

    all_years = sorted(cy_region["year"].unique())

    # -----------------------------------------------------------------------
    # Top: Funnel chart (single selected year)
    # -----------------------------------------------------------------------
    st.subheader("단계별 잔존 건수 (선택 연도)")
    selected_year = st.selectbox("연도", all_years, index=len(all_years) - 1)

    year_data = cy_region[cy_region["year"] == selected_year].copy()

    stages = ["raw", "after_cancel_direct", "after_explicit_excluded", "after_outlier"]
    stage_labels = {
        "raw": "원본",
        "after_cancel_direct": "취소·직거래 제외 후",
        "after_explicit_excluded": "키 결측 제외 후",
        "after_outlier": "이상치 제외 후 (cleaned)",
    }

    funnel_rows: list[dict] = []
    for stage in stages:
        row = year_data[year_data["stage"] == stage]
        if row.empty:
            continue
        count = int(row["count"].iloc[0])
        funnel_rows.append({"stage": stage_labels.get(stage, stage), "count": count})

    if funnel_rows:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Funnel(
                y=[r["stage"] for r in funnel_rows],
                x=[r["count"] for r in funnel_rows],
                textinfo="value+percent initial",
            )
        )
        fig.update_layout(title=f"{selected_year}년 정제 단계별 잔존 건수", height=350)
        st.plotly_chart(fig, width="stretch")

    # -----------------------------------------------------------------------
    # Middle: Stacked area chart (annual trend)
    # -----------------------------------------------------------------------
    st.subheader("연도별 정제 단계 비율 추세")

    pivot = cy_region.pivot_table(
        index="year", columns="stage", values="pct_of_raw", aggfunc="first"
    ).reset_index()

    if not pivot.empty:
        pct_cleaned = pivot.get("after_outlier", pd.Series(dtype=float))
        pct_outlier = (
            pivot.get("after_explicit_excluded", pd.Series(dtype=float)) - pct_cleaned
        ).clip(lower=0)
        pct_exex = (
            pivot.get("after_cancel_direct", pd.Series(dtype=float))
            - pivot.get("after_explicit_excluded", pd.Series(dtype=float))
        ).clip(lower=0)

        # For removed_direct and removed_cancel, use trade_filter_summary
        tf_year = tf_region[tf_region["year"].isin(all_years)].copy() if not tf_region.empty else pd.DataFrame()

        import plotly.graph_objects as go

        fig2 = go.Figure()
        years = pivot["year"].tolist()

        def _pct_series(s: pd.Series) -> list:
            return [float(v) if pd.notna(v) else 0.0 for v in s]

        fig2.add_trace(
            go.Scatter(
                x=years,
                y=_pct_series(pct_cleaned),
                name="cleaned",
                stackgroup="one",
                fillcolor="rgba(46, 134, 193, 0.6)",
                line={"width": 0},
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=years,
                y=_pct_series(pct_outlier),
                name="이상치 제거",
                stackgroup="one",
                fillcolor="rgba(231, 76, 60, 0.5)",
                line={"width": 0},
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=years,
                y=_pct_series(pct_exex),
                name="키 결측 제거",
                stackgroup="one",
                fillcolor="rgba(241, 196, 15, 0.5)",
                line={"width": 0},
            )
        )
        fig2.update_layout(
            title="연도별 정제 단계 비율 (raw=100%)",
            yaxis_title="비율 (%)",
            height=380,
        )
        st.plotly_chart(fig2, width="stretch")

    # -----------------------------------------------------------------------
    # Bottom: Outlier reason breakdown (stacked bar)
    # -----------------------------------------------------------------------
    st.subheader("이상치 유형별 연도 추세")

    rs_region = rs_df[rs_df["region_scope"] == region].copy()
    if not rs_region.empty:
        import plotly.express as px

        reason_pivot = rs_region.groupby(["year", "outlier_reason"])["count"].sum().reset_index()
        fig3 = px.bar(
            reason_pivot,
            x="year",
            y="count",
            color="outlier_reason",
            barmode="stack",
            title="연도별 이상치 유형 분해",
            labels={"count": "건수", "year": "연도", "outlier_reason": "유형"},
            height=350,
            color_discrete_map={
                "sanity_error": "#e74c3c",
                "unsupported_jump": "#e67e22",
                "abs_deviation": "#9b59b6",
                "trend_month_robust_band": "#3498db",
            },
        )
        st.plotly_chart(fig3, width="stretch")
    else:
        st.info("이상치 유형 데이터가 없습니다.")
