"""Level 3 거시지표 연계 분석."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.common import aggregate_rent_scope, aggregate_trade_scope
from analysis.correlation import correlation_matrix, lagged_correlation, simple_regression


def build_scope_frame(
    trade_summary_df: pd.DataFrame,
    rent_summary_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    region_codes: list[str] | None = None,
    scope_name: str | None = None,
) -> pd.DataFrame:
    """매매·전세·거시지표를 하나의 scope 시계열로 결합한다."""
    trade_scope = aggregate_trade_scope(trade_summary_df, region_codes, scope_name)
    rent_scope = aggregate_rent_scope(rent_summary_df, region_codes, scope_name)
    if trade_scope.empty or macro_df.empty:
        return pd.DataFrame()

    jeonse = (
        rent_scope[rent_scope["rentType"] == "전세"][["ym", "date", "평균보증금"]]
        .rename(columns={"평균보증금": "avg_jeonse"})
        if not rent_scope.empty
        else pd.DataFrame(columns=["ym", "date", "avg_jeonse"])
    )

    combined = trade_scope.merge(jeonse, on=["ym", "date"], how="left")
    combined = combined.merge(macro_df, on=["ym", "date"], how="inner")
    combined = combined.sort_values("date").reset_index(drop=True)
    combined["price_yoy"] = combined["평균거래금액"].pct_change(12) * 100
    combined["trade_yoy"] = combined["거래건수"].pct_change(12) * 100
    combined["m2_yoy"] = combined["m2"].pct_change(12) * 100 if "m2" in combined.columns else np.nan
    combined["usdkrw_yoy"] = combined["usdkrw"].pct_change(12) * 100 if "usdkrw" in combined.columns else np.nan
    combined["jeonse_ratio"] = (combined["avg_jeonse"] / combined["평균거래금액"]) * 100
    return combined


def build_rate_lag_chart(combined: pd.DataFrame, scope_name: str) -> go.Figure:
    """금리와 매매가의 시차 상관을 그린다."""
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, subplot_titles=("가격 YoY와 기준금리", "시차 상관계수"))
    if combined.empty or "bok_rate" not in combined.columns:
        fig.update_layout(title="금리 시차 상관 데이터 없음", height=620)
        return fig

    fig.add_trace(go.Scatter(x=combined["date"], y=combined["price_yoy"], name="매매가 YoY", line={"color": "#D1495B"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=combined["date"], y=combined["bok_rate"], name="한국 기준금리", line={"color": "#3E7CB1"}), row=1, col=1)

    lag_source = combined.dropna(subset=["price_yoy", "bok_rate"]).reset_index(drop=True)
    lag_df = lagged_correlation(lag_source, "bok_rate", "price_yoy", max_lag=24) if not lag_source.empty else pd.DataFrame(columns=["lag", "correlation"])
    fig.add_trace(go.Bar(x=lag_df.get("lag", []), y=lag_df.get("correlation", []), marker_color="#2A9D8F"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(title=f"{scope_name} 기준금리 시차 상관", height=640)
    return fig


def build_m2_price_chart(combined: pd.DataFrame, scope_name: str) -> go.Figure:
    """M2와 매매가의 정규화 추이를 이중축으로 표시한다."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if combined.empty or "m2" not in combined.columns:
        fig.update_layout(title="M2 분석 데이터 없음", height=420)
        return fig

    base_price = combined["평균거래금액"].iloc[0]
    base_m2 = combined["m2"].iloc[0]
    if pd.isna(base_price) or pd.isna(base_m2) or base_price == 0 or base_m2 == 0:
        fig.update_layout(title="M2 분석 데이터 없음", height=420)
        return fig

    normalized_price = combined["평균거래금액"] / base_price * 100
    normalized_m2 = combined["m2"] / base_m2 * 100

    fig.add_trace(go.Scatter(x=combined["date"], y=normalized_price, name="매매가 지수", line={"color": "#D1495B"}), secondary_y=False)
    fig.add_trace(go.Scatter(x=combined["date"], y=normalized_m2, name="M2 지수", line={"color": "#457B9D"}), secondary_y=True)
    fig.update_yaxes(title_text="매매가 지수 (기준=100)", secondary_y=False)
    fig.update_yaxes(title_text="M2 지수 (기준=100)", secondary_y=True)
    fig.update_layout(title=f"{scope_name} 매매가 vs M2", height=450, hovermode="x unified")
    return fig


def prepare_fx_event_study(combined: pd.DataFrame, window: int = 6, top_n: int = 3) -> pd.DataFrame:
    """환율 급등 이벤트 전후의 가격 반응을 정렬한다."""
    if combined.empty or "usdkrw" not in combined.columns:
        return pd.DataFrame()

    working = combined.copy()
    working["fx_mom"] = working["usdkrw"].pct_change() * 100
    working["price_mom"] = working["평균거래금액"].pct_change() * 100
    events = working.nlargest(top_n, "fx_mom")["date"].dropna().tolist()

    rows: list[pd.DataFrame] = []
    for event_date in events:
        mask = (working["date"] >= event_date - pd.DateOffset(months=window)) & (working["date"] <= event_date + pd.DateOffset(months=window))
        event_df = working.loc[mask, ["date", "fx_mom", "price_mom"]].copy().sort_values("date")
        if event_df.empty:
            continue
        event_df["event"] = event_date.strftime("%Y-%m")
        event_df["offset"] = list(range(-window, -window + len(event_df)))
        rows.append(event_df)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result["cum_price_mom"] = result.groupby("event", observed=True)["price_mom"].cumsum()
    return result


def build_fx_event_chart(df: pd.DataFrame) -> go.Figure:
    """환율 이벤트 스터디 차트를 생성한다."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="환율 이벤트 데이터 없음", height=420)
        return fig

    for event_name, event_df in df.groupby("event", observed=True):
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


def prepare_real_price_index(combined: pd.DataFrame) -> pd.DataFrame:
    """CPI로 디플레이트한 실질 매매가 지수를 계산한다."""
    if combined.empty or "cpi_kr" not in combined.columns:
        return pd.DataFrame()

    valid_cpi = combined["cpi_kr"].dropna()
    if valid_cpi.empty or combined["평균거래금액"].dropna().empty:
        return pd.DataFrame()

    base_cpi = valid_cpi.iloc[0]
    base_price = combined["평균거래금액"].iloc[0]
    if base_cpi == 0 or pd.isna(base_price) or base_price == 0:
        return pd.DataFrame()

    real_price = combined["평균거래금액"] / (combined["cpi_kr"] / base_cpi)
    result = combined[["date", "평균거래금액"]].copy()
    result["nominal_index"] = result["평균거래금액"] / base_price * 100
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


def prepare_combined_correlation(combined: pd.DataFrame) -> pd.DataFrame:
    """매매가, 전세가, 거시지표를 합친 상관분석용 프레임을 만든다."""
    if combined.empty:
        return combined.copy()

    result = combined.rename(columns={"평균거래금액": "avg_trade_price", "거래건수": "trade_count"}).copy()
    desired_columns = [
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
    ]
    available = [column for column in desired_columns if column in result.columns]
    return result[available].dropna(how="all")


def build_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """단일 상관계수 히트맵을 생성한다."""
    numeric = df.select_dtypes(include=[np.number]) if not df.empty else pd.DataFrame()
    if numeric.empty:
        fig = go.Figure()
        fig.update_layout(title="복합 상관계수 데이터 없음")
        return fig

    corr = correlation_matrix(numeric)
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
    numeric_a = df_a.select_dtypes(include=[np.number]) if not df_a.empty else pd.DataFrame()
    numeric_b = df_b.select_dtypes(include=[np.number]) if not df_b.empty else pd.DataFrame()
    if numeric_a.empty or numeric_b.empty:
        fig = go.Figure()
        fig.update_layout(title="비교할 상관 구조 데이터가 없습니다.")
        return fig

    corr_a = correlation_matrix(numeric_a)
    corr_b = correlation_matrix(numeric_b)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(label_a, label_b))
    fig.add_trace(go.Heatmap(z=corr_a.values, x=corr_a.columns, y=corr_a.index, zmin=-1, zmax=1, colorscale="RdBu_r"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=corr_b.values, x=corr_b.columns, y=corr_b.index, zmin=-1, zmax=1, colorscale="RdBu_r", showscale=False), row=1, col=2)
    fig.update_layout(height=620, title="지역별 상관 구조 비교")
    return fig


def build_macro_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[go.Figure, dict[str, float]]:
    """선형 회귀선이 포함된 산점도를 생성한다."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="산점도 데이터 없음")
        return fig, {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan}

    reg = simple_regression(df, x_col, y_col)
    subset = df[[x_col, y_col]].dropna()
    fig = px.scatter(subset, x=x_col, y=y_col, trendline="ols" if len(subset) >= 3 else None, title=f"{x_col} vs {y_col}")
    return fig, reg