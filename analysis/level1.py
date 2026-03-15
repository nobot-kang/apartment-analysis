"""Level 1 기초 현황 분석."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.common import (
    AGE_LABELS,
    AREA_BINS,
    AREA_LABELS,
    POLICY_EVENTS,
    aggregate_rent_scope,
    aggregate_trade_scope,
    classify_age,
)

RISK_THRESHOLD = 80.0


def _add_datetime_event_marker(fig: go.Figure, event_date: pd.Timestamp, label: str) -> None:
    """Plotly datetime 축의 add_vline 주석 버그를 우회해 이벤트 마커를 추가한다."""
    fig.add_vline(x=event_date, line_dash="dash", line_color="gray")
    fig.add_annotation(
        x=event_date,
        y=1,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "gray"},
        bgcolor="rgba(255,255,255,0.75)",
    )


def build_overview_metrics(
    trade_df: pd.DataFrame,
    rent_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> dict[str, float]:
    """Level 1 상단 KPI를 계산한다."""
    if trade_df.empty:
        return {
            "latest_avg_trade": float("nan"),
            "latest_trade_count": float("nan"),
            "latest_ratio": float("nan"),
            "latest_rate": float("nan"),
            "latest_ym": "N/A",
        }

    latest_ym = str(trade_df["ym"].max())
    latest_trade = trade_df[trade_df["ym"] == latest_ym].copy()
    latest_rent = rent_df[(rent_df["ym"] == latest_ym) & (rent_df["rentType"] == "전세")].copy() if not rent_df.empty else pd.DataFrame()

    latest_ratio = float("nan")
    if not latest_trade.empty and not latest_rent.empty:
        latest_ratio = float(latest_rent["평균보증금"].mean() / latest_trade["평균거래금액"].mean() * 100)

    latest_rate = float("nan")
    if not macro_df.empty and "bok_rate" in macro_df.columns:
        valid_rates = macro_df["bok_rate"].dropna()
        if not valid_rates.empty:
            latest_rate = float(valid_rates.iloc[-1])

    return {
        "latest_avg_trade": float(latest_trade["평균거래금액"].mean()) if not latest_trade.empty else float("nan"),
        "latest_trade_count": float(latest_trade["거래건수"].sum()) if not latest_trade.empty else float("nan"),
        "latest_ratio": latest_ratio,
        "latest_rate": latest_rate,
        "latest_ym": latest_ym,
    }


def build_monthly_volume_frame(
    trade_df: pd.DataFrame,
    rent_df: pd.DataFrame,
    region_codes: list[str] | None = None,
    scope_name: str | None = None,
) -> pd.DataFrame:
    """매매·전세·월세 거래량을 long format으로 정리한다."""
    trade = aggregate_trade_scope(trade_df, region_codes, scope_name)
    rent = aggregate_rent_scope(rent_df, region_codes, scope_name)

    trade_long = trade[["ym", "date", "거래건수", "scope_name"]].copy() if not trade.empty else pd.DataFrame()
    if not trade_long.empty:
        trade_long["deal_type"] = "매매"
        trade_long = trade_long.rename(columns={"거래건수": "count"})

    rent_long = rent[["ym", "date", "rentType", "거래건수", "scope_name"]].copy() if not rent.empty else pd.DataFrame()
    if not rent_long.empty:
        rent_long = rent_long.rename(columns={"rentType": "deal_type", "거래건수": "count"})

    if trade_long.empty and rent_long.empty:
        return pd.DataFrame(columns=["ym", "date", "deal_type", "count", "scope_name"])

    return pd.concat([trade_long, rent_long], ignore_index=True).sort_values(["date", "deal_type"]).reset_index(drop=True)


def build_monthly_volume_chart(df: pd.DataFrame, scope_name: str, highlight_events: bool = True) -> go.Figure:
    """월별 거래량 추이와 매매 거래량 막대를 함께 표시한다."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("거래 유형별 월별 거래량", "매매 거래량"),
    )

    if df.empty:
        fig.update_layout(title=f"{scope_name} 거래량 데이터 없음", height=620)
        return fig

    colors = {"매매": "#D1495B", "전세": "#3E7CB1", "월세": "#2A9D8F"}
    for deal_type in ["매매", "전세", "월세"]:
        subset = df[df["deal_type"] == deal_type].copy()
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["count"],
                mode="lines+markers",
                name=deal_type,
                line={"width": 2.5, "color": colors.get(deal_type, "#666666")},
                marker={"size": 5},
            ),
            row=1,
            col=1,
        )

    trade_only = df[df["deal_type"] == "매매"].copy()
    if not trade_only.empty:
        fig.add_trace(
            go.Bar(
                x=trade_only["date"],
                y=trade_only["count"],
                marker_color=colors["매매"],
                opacity=0.7,
                name="매매 거래량",
            ),
            row=2,
            col=1,
        )

    if highlight_events:
        for date_str, label in POLICY_EVENTS.items():
            _add_datetime_event_marker(fig, pd.Timestamp(date_str), label)

    fig.update_layout(
        title=f"{scope_name} 거래량 추이",
        height=620,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.05},
    )
    fig.update_yaxes(title_text="거래건수", row=1, col=1)
    fig.update_yaxes(title_text="매매 거래건수", row=2, col=1)
    return fig


def filter_district_ranking(yearly_metrics_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """연도별 지역 랭킹 집계를 필터링한다."""
    if yearly_metrics_df.empty:
        return pd.DataFrame(columns=["_lawd_cd", "_region_name", "avg_price", "avg_price_per_m2", "trade_count"])

    result = yearly_metrics_df[yearly_metrics_df["year"] == year].copy()
    if result.empty:
        return pd.DataFrame(columns=["_lawd_cd", "_region_name", "avg_price", "avg_price_per_m2", "trade_count"])
    return result.sort_values("avg_price", ascending=True).reset_index(drop=True)


def build_ranking_chart(df: pd.DataFrame, year: int, metric: str = "avg_price") -> go.Figure:
    """연도별 지역 랭킹 수평 바차트를 그린다."""
    label_map = {
        "avg_price": "평균 매매가 (만원)",
        "avg_price_per_m2": "평균 ㎡당 가격 (만원)",
    }
    if df.empty:
        return px.bar(title=f"{year}년 랭킹 데이터가 없습니다.")

    metric_label = label_map.get(metric, metric)
    sorted_df = df.sort_values(metric, ascending=True).copy()
    fig = px.bar(
        sorted_df,
        x=metric,
        y="_region_name",
        orientation="h",
        color=metric,
        color_continuous_scale="RdYlGn_r",
        text=sorted_df[metric].map(lambda value: f"{value:,.0f}"),
        labels={"_region_name": "지역", metric: metric_label},
        title=f"{year}년 지역별 {metric_label} 랭킹",
        hover_data={"trade_count": True, "_lawd_cd": False},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=900, coloraxis_showscale=False)
    return fig


def build_ranking_animation(yearly_metrics_df: pd.DataFrame, metric: str = "avg_price") -> go.Figure:
    """연도별 지역 랭킹 변화를 애니메이션 바차트로 생성한다."""
    if yearly_metrics_df.empty:
        return px.bar(title="랭킹 데이터가 없습니다.")

    yearly = yearly_metrics_df.copy()
    yearly["year_label"] = yearly["year"].astype(int).astype(str)
    max_value = yearly[metric].max(skipna=True)
    if pd.isna(max_value) or max_value <= 0:
        max_value = 1.0

    fig = px.bar(
        yearly.sort_values(["year", metric], ascending=[True, True]),
        x=metric,
        y="_region_name",
        color=metric,
        orientation="h",
        animation_frame="year_label",
        color_continuous_scale="RdYlGn_r",
        range_x=[0, float(max_value) * 1.1],
        labels={
            "_region_name": "지역",
            "avg_price": "평균 매매가 (만원)",
            "avg_price_per_m2": "평균 ㎡당 가격 (만원)",
        },
        title="연도별 지역 랭킹 변화",
        hover_data={"trade_count": True},
    )
    fig.update_layout(height=900)
    return fig


def prepare_area_distribution(trade_detail_df: pd.DataFrame) -> pd.DataFrame:
    """면적 구간별 가격 분포 분석용 상세 데이터를 정리한다."""
    if trade_detail_df.empty:
        return trade_detail_df.copy()

    result = trade_detail_df.copy()
    result["area_bin"] = pd.cut(result["area"], bins=AREA_BINS, labels=AREA_LABELS, right=False)
    return result.dropna(subset=["area_bin", "price"])


def build_area_boxplot(df: pd.DataFrame, area_bin: str, scope_name: str) -> go.Figure:
    """선택 면적 구간의 연도별 가격 박스플롯을 생성한다."""
    fig = go.Figure()
    subset = df[df["area_bin"] == area_bin].copy()
    if subset.empty:
        fig.update_layout(title=f"{scope_name} {area_bin} 데이터 없음", height=480)
        return fig

    low, high = subset["price"].quantile([0.01, 0.99])
    subset = subset[subset["price"].between(low, high)].copy()
    for year in sorted(subset["year"].dropna().unique()):
        fig.add_trace(
            go.Box(
                y=subset.loc[subset["year"] == year, "price"],
                name=str(int(year)),
                boxmean="sd",
            )
        )

    fig.update_layout(
        title=f"{scope_name} {area_bin} 가격 분포",
        xaxis_title="연도",
        yaxis_title="거래금액 (만원)",
        height=500,
        showlegend=False,
    )
    return fig


def prepare_age_premium(trade_detail_df: pd.DataFrame) -> pd.DataFrame:
    """건축 연령 프리미엄 분석용 상세 데이터를 생성한다."""
    if trade_detail_df.empty:
        return trade_detail_df.copy()

    result = trade_detail_df.copy()
    result["price_per_m2"] = result["price"] / result["area"].replace(0, pd.NA)
    result["age_bin"] = result["age"].apply(classify_age)
    return result.dropna(subset=["price_per_m2"])


def build_age_premium_chart(df: pd.DataFrame, region_name: str, year: int) -> go.Figure:
    """선택 지역/연도의 건축 연령별 가격 프리미엄 바차트."""
    subset = df[(df["region_name"] == region_name) & (df["year"] == year)].copy()
    if subset.empty:
        return px.bar(title=f"{region_name} {year}년 데이터가 없습니다.")

    aggregated = (
        subset.groupby("age_bin", observed=True)["price_per_m2"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    ordered_bins = AGE_LABELS + ["미상"]
    aggregated["age_bin"] = pd.Categorical(aggregated["age_bin"], categories=ordered_bins, ordered=True)
    aggregated = aggregated.sort_values("age_bin")

    fig = px.bar(
        aggregated,
        x="age_bin",
        y="mean",
        color="mean",
        color_continuous_scale="Blues",
        text=aggregated["mean"].map(lambda value: f"{value:,.0f}"),
        labels={"age_bin": "건축 연령", "mean": "평균 ㎡당 가격 (만원)"},
        title=f"{region_name} {year}년 건축 연령 프리미엄",
        hover_data={"median": ":.0f", "count": True},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=460, coloraxis_showscale=False)
    return fig


def build_jeonse_ratio_chart(df: pd.DataFrame, region_name: str) -> go.Figure:
    """특정 지역의 전세가율 추이를 시각화한다."""
    fig = go.Figure()
    subset = df[df["_region_name"] == region_name].copy()
    if subset.empty:
        fig.update_layout(title=f"{region_name} 전세가율 데이터 없음", height=460)
        return fig

    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=subset["전세가율"],
            mode="lines+markers",
            line={"width": 2.5, "color": "#2C7FB8"},
            name="전세가율",
        )
    )
    fig.add_hline(
        y=RISK_THRESHOLD,
        line_dash="dash",
        line_color="#D1495B",
        annotation_text="위험 기준 80%",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"{region_name} 전세가율 추이",
        yaxis_title="전세가율 (%)",
        height=460,
        hovermode="x unified",
    )
    return fig