"""Level 3 거시지표 연계 분석."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.common import ANALYSIS_START_YM, aggregate_rent_scope, aggregate_trade_scope, load_macro_monthly_df, load_rent_summary_df, load_trade_summary_df
from analysis.correlation import correlation_matrix, lagged_correlation, simple_regression


def _build_scope_frame(region_codes: list[str] | None = None, scope_name: str | None = None) -> pd.DataFrame:
    trade_scope = aggregate_trade_scope(load_trade_summary_df(ANALYSIS_START_YM), region_codes, scope_name)
    rent_scope = aggregate_rent_scope(load_rent_summary_df(ANALYSIS_START_YM), region_codes, scope_name)
    macro = load_macro_monthly_df(ANALYSIS_START_YM)
    if trade_scope.empty or macro.empty:
        return pd.DataFrame()

    jeonse = rent_scope[rent_scope["rentType"] == "전세"][["ym", "date", "평균보증금"]].rename(columns={"평균보증금": "avg_jeonse"}) if not rent_scope.empty else pd.DataFrame()
    combined = trade_scope.merge(jeonse, on=["ym", "date"], how="left")
    combined = combined.merge(macro, on=["ym", "date"], how="inner")
    combined["price_yoy"] = combined["평균거래금액"].pct_change(12) * 100
    combined["trade_yoy"] = combined["거래건수"].pct_change(12) * 100
    combined["m2_yoy"] = combined["m2"].pct_change(12) * 100 if "m2" in combined.columns else np.nan
    combined["usdkrw_yoy"] = combined["usdkrw"].pct_change(12) * 100 if "usdkrw" in combined.columns else np.nan
    combined["jeonse_ratio"] = (combined["avg_jeonse"] / combined["평균거래금액"]) * 100
    return combined.sort_values("date").reset_index(drop=True)


def build_rate_lag_chart(region_codes: list[str] | None = None, scope_name: str | None = None) -> go.Figure:
    """금리와 매매가의 시차 상관을 그린다."""
    combined = _build_scope_frame(region_codes, scope_name)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, subplot_titles=("가격 YoY와 기준금리", "시차 상관계수"))
    if combined.empty:
        fig.update_layout(title="금리 시차 상관 데이터 없음", height=620)
        return fig

    fig.add_trace(go.Scatter(x=combined["date"], y=combined["price_yoy"], name="매매가 YoY", line={"color": "#D1495B"}), row=1, col=1)
    if "bok_rate" in combined.columns:
        fig.add_trace(go.Scatter(x=combined["date"], y=combined["bok_rate"], name="한국 기준금리", line={"color": "#3E7CB1"}), row=1, col=1)

    lag_df = lagged_correlation(combined.dropna(subset=["price_yoy", "bok_rate"]), "bok_rate", "price_yoy", max_lag=24)
    fig.add_trace(go.Bar(x=lag_df["lag"], y=lag_df["correlation"], marker_color="#2A9D8F"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(title=f"{scope_name or '선택 지역'} 기준금리 시차 상관", height=640)
    return fig


def build_m2_price_chart(region_codes: list[str] | None = None, scope_name: str | None = None) -> go.Figure:
    """M2와 매매가의 정규화 추이를 이중축으로 표시한다."""
    combined = _build_scope_frame(region_codes, scope_name)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if combined.empty or "m2" not in combined.columns:
        fig.update_layout(title="M2 분석 데이터 없음", height=420)
        return fig

    normalized_price = combined["평균거래금액"] / combined["평균거래금액"].iloc[0] * 100
    normalized_m2 = combined["m2"] / combined["m2"].iloc[0] * 100

    fig.add_trace(go.Scatter(x=combined["date"], y=normalized_price, name="매매가 지수", line={"color": "#D1495B"}), secondary_y=False)
    fig.add_trace(go.Scatter(x=combined["date"], y=normalized_m2, name="M2 지수", line={"color": "#457B9D"}), secondary_y=True)
    fig.update_yaxes(title_text="매매가 지수 (기준=100)", secondary_y=False)
    fig.update_yaxes(title_text="M2 지수 (기준=100)", secondary_y=True)
    fig.update_layout(title=f"{scope_name or '선택 지역'} 매매가 vs M2", height=450, hovermode="x unified")
    return fig


def load_fx_event_study(window: int = 6, top_n: int = 3) -> pd.DataFrame:
    """환율 급등 이벤트 전후의 가격 반응을 정렬한다."""
    combined = _build_scope_frame(list(load_trade_summary_df(ANALYSIS_START_YM)["_lawd_cd"].astype(str).unique()), "수도권 전체")
    if combined.empty or "usdkrw" not in combined.columns:
        return pd.DataFrame()

    combined = combined.copy()
    combined["fx_mom"] = combined["usdkrw"].pct_change() * 100
    combined["price_mom"] = combined["평균거래금액"].pct_change() * 100
    events = combined.nlargest(top_n, "fx_mom")["date"].tolist()

    rows = []
    for event_date in events:
        window_mask = (combined["date"] >= event_date - pd.DateOffset(months=window)) & (combined["date"] <= event_date + pd.DateOffset(months=window))
        event_df = combined.loc[window_mask, ["date", "fx_mom", "price_mom"]].copy().sort_values("date")
        event_df["event"] = event_date.strftime("%Y-%m")
        event_df["offset"] = range(-window, -window + len(event_df))
        rows.append(event_df)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result["cum_price_mom"] = result.groupby("event")["price_mom"].cumsum()
    return result


def build_fx_event_chart(df: pd.DataFrame) -> go.Figure:
    """환율 이벤트 스터디 차트를 생성한다."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="환율 이벤트 데이터 없음", height=420)
        return fig

    for event_name, event_df in df.groupby("event"):
        fig.add_trace(
            go.Scatter(
                x=event_df["offset"],
                y=event_df["cum_price_mom"],
                mode="lines+markers",
                name=event_name,
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="환율 급등 이벤트 전후 누적 가격 반응", xaxis_title="이벤트 기준 월", yaxis_title="누적 가격 MoM (%)", height=430)
    return fig


def load_real_price_index(region_codes: list[str] | None = None, scope_name: str | None = None) -> pd.DataFrame:
    """CPI로 디플레이트한 실질 매매가 지수를 계산한다."""
    combined = _build_scope_frame(region_codes, scope_name)
    if combined.empty or "cpi_kr" not in combined.columns:
        return pd.DataFrame()

    base_cpi = combined["cpi_kr"].dropna().iloc[0]
    real_price = combined["평균거래금액"] / (combined["cpi_kr"] / base_cpi)
    result = combined[["date", "평균거래금액"]].copy()
    result["nominal_index"] = result["평균거래금액"] / result["평균거래금액"].iloc[0] * 100
    result["real_index"] = real_price / real_price.iloc[0] * 100
    return result


def build_real_price_chart(df: pd.DataFrame, scope_name: str) -> go.Figure:
    """명목/실질 가격 지수를 한 차트에 그린다."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="실질 가격 지수 데이터 없음", height=420)
        return fig

    fig.add_trace(go.Scatter(x=df["date"], y=df["nominal_index"], name="명목 지수", line={"color": "#D1495B"}))
    fig.add_trace(go.Scatter(x=df["date"], y=df["real_index"], name="실질 지수", line={"color": "#457B9D"}))
    fig.update_layout(title=f"{scope_name} 명목 vs 실질 매매가 지수", yaxis_title="지수 (기준=100)", height=430, hovermode="x unified")
    return fig


def load_combined_correlation(region_codes: list[str] | None = None, scope_name: str | None = None) -> pd.DataFrame:
    """매매가, 전세가, 거시지표를 합친 상관분석용 프레임을 만든다."""
    combined = _build_scope_frame(region_codes, scope_name)
    if combined.empty:
        return combined

    result = combined.copy()
    result = result.rename(columns={"평균거래금액": "avg_trade_price", "거래건수": "trade_count"})
    return result[[
        "date",
        "avg_trade_price",
        "avg_jeonse",
        "jeonse_ratio",
        "trade_count",
        "bok_rate",
        "fed_rate",
        "cpi_kr",
        "cpi_us",
        "m2",
        "gold",
        "oil",
        "usdkrw",
    ]].dropna(how="all")


def build_correlation_heatmap(df: pd.DataFrame) -> px.imshow:
    """단일 상관계수 히트맵을 생성한다."""
    corr = correlation_matrix(df)
    fig = px.imshow(
        corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        title="복합 상관계수 히트맵",
        aspect="auto",
    )
    fig.update_layout(height=620)
    return fig


def build_dual_correlation_heatmaps(df_a: pd.DataFrame, label_a: str, df_b: pd.DataFrame, label_b: str) -> go.Figure:
    """두 지역의 상관계수 행렬을 나란히 비교한다."""
    corr_a = correlation_matrix(df_a)
    corr_b = correlation_matrix(df_b)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(label_a, label_b))
    fig.add_trace(go.Heatmap(z=corr_a.values, x=corr_a.columns, y=corr_a.index, zmin=-1, zmax=1, colorscale="RdBu_r"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=corr_b.values, x=corr_b.columns, y=corr_b.index, zmin=-1, zmax=1, colorscale="RdBu_r", showscale=False), row=1, col=2)
    fig.update_layout(height=620, title="지역별 상관 구조 비교")
    return fig


def build_macro_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[px.scatter, dict[str, float]]:
    """선형 회귀선이 포함된 산점도를 생성한다."""
    reg = simple_regression(df, x_col, y_col)
    subset = df[[x_col, y_col]].dropna()
    fig = px.scatter(subset, x=x_col, y=y_col, trendline="ols" if len(subset) >= 3 else None, title=f"{x_col} vs {y_col}")
    return fig, reg
