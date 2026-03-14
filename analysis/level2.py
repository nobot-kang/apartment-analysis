"""Level 2 심화 비교 분석."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.common import (
    ANALYSIS_START_YM,
    FLOOR_BINS,
    FLOOR_LABELS,
    add_seoul_coordinates,
    aggregate_rent_scope,
    aggregate_trade_scope,
    load_rent_summary_df,
    load_trade_detail_df,
    load_trade_summary_df,
)
from analysis.correlation import lagged_correlation


def _aggregate_yearly_trade(trade: pd.DataFrame, include_price_per_m2: bool = True) -> pd.DataFrame:
    """연도/지역 단위 가중 평균 매매 지표를 계산한다."""
    if trade.empty:
        columns = ["_lawd_cd", "_region_name", "year", "avg_price", "trade_count"]
        if include_price_per_m2:
            columns.append("avg_price_per_m2")
        return pd.DataFrame(columns=columns)

    base_columns = ["_lawd_cd", "_region_name", "year", "평균거래금액", "거래건수"]
    if include_price_per_m2:
        base_columns.append("평균거래금액_전용면적당")
    working = trade[base_columns].copy()
    working["weighted_price"] = working["평균거래금액"] * working["거래건수"]
    if include_price_per_m2:
        working["weighted_price_per_m2"] = working["평균거래금액_전용면적당"] * working["거래건수"]

    agg_spec: dict[str, tuple[str, str]] = {
        "weighted_price": ("weighted_price", "sum"),
        "trade_count": ("거래건수", "sum"),
    }
    if include_price_per_m2:
        agg_spec["weighted_price_per_m2"] = ("weighted_price_per_m2", "sum")

    yearly = (
        working.groupby(["_lawd_cd", "_region_name", "year"], observed=True)
        .agg(**agg_spec)
        .reset_index()
    )
    yearly = yearly[yearly["trade_count"] > 0].copy()
    yearly["avg_price"] = yearly["weighted_price"] / yearly["trade_count"]
    if include_price_per_m2:
        yearly["avg_price_per_m2"] = yearly["weighted_price_per_m2"] / yearly["trade_count"]
    return yearly


def build_district_year_heatmap(metric: str = "avg_price") -> go.Figure:
    """지역별 연간 평균 가격 또는 상승률 히트맵을 생성한다."""
    trade = load_trade_summary_df(ANALYSIS_START_YM)
    if trade.empty:
        return go.Figure()

    yearly = _aggregate_yearly_trade(trade, include_price_per_m2=True)
    value_col = "avg_price"
    if metric == "avg_price_per_m2":
        value_col = "avg_price_per_m2"

    pivot = yearly.pivot(index="_region_name", columns="year", values=value_col).sort_index()
    if metric == "yoy_change":
        pivot = pivot.pct_change(axis=1) * 100

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(value) for value in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r" if metric != "yoy_change" else "RdBu_r",
            zmid=0 if metric == "yoy_change" else None,
            text=[[f"{value:,.1f}" if pd.notna(value) else "" for value in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    title_map = {
        "avg_price": "지역별 연간 평균 매매가",
        "avg_price_per_m2": "지역별 연간 평균 ㎡당 가격",
        "yoy_change": "지역별 연간 YoY 상승률 (%)",
    }
    fig.update_layout(
        title=title_map.get(metric, metric),
        height=900,
        xaxis_title="연도",
        yaxis_title="지역",
    )
    return fig


def load_floor_premium_data(region_codes: list[str] | None = None, years: list[int] | None = None) -> pd.DataFrame:
    """층수 프리미엄 비교용 상세 데이터를 만든다."""
    df = load_trade_detail_df(
        years=years,
        region_codes=region_codes,
        columns=["date", "price", "area", "floor", "dong_repr"],
    )
    if df.empty:
        return df

    result = df.copy()
    result["price_per_m2"] = result["price"] / result["area"].replace(0, pd.NA)
    result["floor_bin"] = pd.cut(result["floor"], bins=FLOOR_BINS, labels=FLOOR_LABELS, right=True)
    return result.dropna(subset=["price_per_m2", "floor_bin"])


def build_floor_premium_chart(df: pd.DataFrame, compare_regions: list[str], year: int) -> go.Figure:
    """선택 지역들의 층수 프리미엄을 비교한다."""
    subset = df[(df["region_name"].isin(compare_regions)) & (df["year"] == year)].copy()
    if subset.empty:
        return px.bar(title="층수 프리미엄 데이터가 없습니다.")

    aggregated = (
        subset.groupby(["region_name", "floor_bin"], observed=True)["price_per_m2"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    aggregated["floor_bin"] = pd.Categorical(aggregated["floor_bin"], categories=FLOOR_LABELS, ordered=True)
    aggregated = aggregated.sort_values(["region_name", "floor_bin"])

    fig = px.bar(
        aggregated,
        x="floor_bin",
        y="mean",
        color="region_name",
        barmode="group",
        text=aggregated["mean"].map(lambda value: f"{value:,.0f}"),
        labels={"floor_bin": "층수 구간", "mean": "평균 ㎡당 가격 (만원)", "region_name": "지역"},
        title=f"{year}년 층수 프리미엄 비교",
        hover_data={"median": ":.0f", "count": True},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=520)
    return fig


def load_yoy_map_data(target_year: int) -> pd.DataFrame:
    """서울 자치구의 연간 평균 매매가 YoY 상승률 데이터를 준비한다."""
    trade = load_trade_summary_df(ANALYSIS_START_YM)
    if trade.empty:
        return pd.DataFrame()

    seoul = trade[trade["_lawd_cd"].astype(str).str.startswith("11")].copy()
    yearly = _aggregate_yearly_trade(seoul, include_price_per_m2=False)

    current_year = yearly[yearly["year"] == target_year].copy()
    previous_year = yearly[yearly["year"] == target_year - 1][["_lawd_cd", "avg_price"]].rename(columns={"avg_price": "prev_price"})
    yoy = current_year.merge(previous_year, on="_lawd_cd", how="left")
    yoy["yoy_pct"] = (yoy["avg_price"] / yoy["prev_price"] - 1) * 100
    yoy = add_seoul_coordinates(yoy)
    return yoy.dropna(subset=["lat", "lon", "yoy_pct"]).reset_index(drop=True)


def build_yoy_map(yoy_df: pd.DataFrame, target_year: int) -> go.Figure:
    """서울 자치구 YoY 상승률 버블맵을 표시한다."""
    if yoy_df.empty:
        return go.Figure()

    fig = px.scatter_mapbox(
        yoy_df,
        lat="lat",
        lon="lon",
        color="yoy_pct",
        size="trade_count",
        hover_name="_region_name",
        hover_data={"avg_price": ":,.0f", "trade_count": True, "yoy_pct": ":.2f", "lat": False, "lon": False},
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zoom=10,
        center={"lat": 37.5665, "lon": 126.9780},
        mapbox_style="open-street-map",
        title=f"{target_year}년 서울 자치구 YoY 상승률 지도",
    )
    fig.update_layout(height=620, margin={"l": 0, "r": 0, "t": 50, "b": 0})
    return fig


def load_volume_price_lag_data(region_codes: list[str] | None = None, scope_name: str | None = None) -> pd.DataFrame:
    """거래량과 가격의 선후행 관계 분석용 월별 시계열을 만든다."""
    trade_scope = aggregate_trade_scope(load_trade_summary_df(ANALYSIS_START_YM), region_codes, scope_name)
    if trade_scope.empty:
        return pd.DataFrame()

    result = trade_scope.copy()
    result["price_yoy"] = result["평균거래금액"].pct_change(12) * 100
    result["volume_yoy"] = result["거래건수"].pct_change(12) * 100
    return result.dropna(subset=["price_yoy", "volume_yoy"])


def build_volume_price_lag_chart(df: pd.DataFrame, scope_name: str) -> go.Figure:
    """거래량과 가격의 선행·후행 관계를 서브플롯으로 보여준다."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=("거래량 YoY vs 매매가 YoY", "시차 상관계수"),
    )
    if df.empty:
        fig.update_layout(title=f"{scope_name} 선후행 데이터 없음", height=620)
        return fig

    fig.add_trace(
        go.Scatter(x=df["date"], y=df["price_yoy"], name="가격 YoY", line={"color": "#D1495B", "width": 2.5}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["volume_yoy"], name="거래량 YoY", line={"color": "#3E7CB1", "width": 2.5}),
        row=1,
        col=1,
    )

    lag_df = lagged_correlation(df, "volume_yoy", "price_yoy", max_lag=12)
    fig.add_trace(
        go.Bar(x=lag_df["lag"], y=lag_df["correlation"], marker_color="#2A9D8F", name="lag corr"),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(title=f"{scope_name} 거래량-가격 선후행 분석", height=640, hovermode="x unified")
    fig.update_xaxes(title_text="기준월", row=1, col=1)
    fig.update_xaxes(title_text="거래량 선행 개월수(+는 거래량 선행)", row=2, col=1)
    return fig


def load_conversion_rate_data(region_codes: list[str] | None = None, scope_name: str | None = None) -> pd.DataFrame:
    """전월세 전환율을 역산한다."""
    rent_scope = aggregate_rent_scope(load_rent_summary_df(ANALYSIS_START_YM), region_codes, scope_name)
    if rent_scope.empty:
        return pd.DataFrame()

    jeonse = rent_scope[rent_scope["rentType"] == "전세"]["ym date 평균보증금 scope_name".split()].rename(columns={"평균보증금": "jeonse_deposit"})
    wolse = rent_scope[rent_scope["rentType"] == "월세"]["ym date 평균보증금 평균월세 거래건수 scope_name".split()].rename(columns={"평균보증금": "wolse_deposit", "거래건수": "sample_count"})
    merged = wolse.merge(jeonse, on=["ym", "date", "scope_name"], how="left")
    merged["deposit_gap"] = merged["jeonse_deposit"] - merged["wolse_deposit"]
    merged = merged[merged["deposit_gap"] > 0].copy()
    merged["conversion_rate"] = (merged["평균월세"] * 12 / merged["deposit_gap"]) * 100
    merged = merged[(merged["conversion_rate"] >= 0) & (merged["conversion_rate"] <= 30)]
    return merged.sort_values("date").reset_index(drop=True)


def build_conversion_rate_chart(df: pd.DataFrame, scope_name: str) -> go.Figure:
    """전월세 전환율 시계열 차트를 생성한다."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=f"{scope_name} 전월세 전환율 데이터 없음", height=420)
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["conversion_rate"],
            mode="lines+markers",
            line={"color": "#5E548E", "width": 2.5},
            name="전환율",
        )
    )
    fig.update_layout(
        title=f"{scope_name} 전월세 전환율 추이",
        yaxis_title="전환율 (%)",
        height=430,
        hovermode="x unified",
    )
    return fig
