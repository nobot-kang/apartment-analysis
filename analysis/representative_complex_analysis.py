"""Representative-complex analytics for 59-type and 84-type price panels."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots

from analysis.complex_analysis import (
    ModelResult,
    _empty_figure,
    _numeric,
    _safe_qcut,
    build_effect_chart,
    build_forecast_chart,
    build_importance_chart,
)

AREA_BAND_LABELS = {59: "59형", 84: "84형"}
MARKET_TYPE_LABELS = {"sale": "매매", "jeonse": "전세", "wolse": "월세"}
FEATURE_LABELS = {
    "log_households": "로그 세대수",
    "parking_per_household": "세대당 주차대수",
    "floor_area_ratio": "용적률",
    "building_coverage_ratio": "건폐율",
    "avg_land_area_per_household": "세대당 대지면적",
    "redevelopment_option_score": "재건축 옵션 점수",
    "bok_rate_change_3m": "기준금리 3개월 변화",
    "m2_yoy": "M2 YoY",
    "usdkrw_change_3m": "환율 3개월 변화",
    "rate_x_parking": "금리 x 주차",
    "rate_x_density": "금리 x 저밀도",
    "liquidity_x_redevelop": "유동성 x 재건축",
    "sale_gap_ratio_lag1": "직전 1개월 gap",
    "sale_gap_ratio_lag3": "직전 3개월 gap",
    "sale_py_59_roll3_lag1": "59형 직전 1개월",
    "sale_py_84_roll3_lag1": "84형 직전 1개월",
    "sale_59_fill_age": "59형 관측공백",
    "sale_84_fill_age": "84형 관측공백",
    "sale_any_imputed": "보간 여부",
}
BASE_FEATURES = [
    "log_households",
    "parking_per_household",
    "floor_area_ratio",
    "building_coverage_ratio",
    "avg_land_area_per_household",
    "redevelopment_option_score",
]
FORECAST_FEATURES = [
    "log_households",
    "parking_per_household",
    "floor_area_ratio",
    "building_coverage_ratio",
    "avg_land_area_per_household",
    "redevelopment_option_score",
    "sale_py_59_roll3_lag1",
    "sale_py_59_roll3_lag3",
    "sale_py_84_roll3_lag1",
    "sale_py_84_roll3_lag3",
    "sale_gap_ratio_lag1",
    "sale_gap_ratio_lag3",
    "sale_59_fill_age",
    "sale_84_fill_age",
    "sale_any_imputed",
    "bok_rate",
    "bok_rate_change_3m",
    "m2_yoy",
    "usdkrw",
]


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result


def _ensure_region_name(df: pd.DataFrame, region_level: str) -> pd.DataFrame:
    result = df.copy()
    if region_level == "sigungu":
        result["region_label"] = result.get("sigungu_name", result.get("region_name", result.get("sigungu_code")))
        result["region_key"] = result.get("sigungu_code", result.get("region_code"))
    else:
        result["region_label"] = result.get("dong_repr", result.get("region_name", result.get("region_code")))
        result["region_key"] = result.get("dong_repr", result.get("region_code"))
    return result


def get_region_option_map(df: pd.DataFrame, region_level: str) -> dict[str, str]:
    if df.empty:
        return {}
    working = _ensure_region_name(df, region_level)
    options = (
        working[["region_key", "region_label"]]
        .dropna()
        .drop_duplicates()
        .sort_values("region_label")
    )
    return {str(row["region_label"]): str(row["region_key"]) for _, row in options.iterrows()}


def list_complex_options(universe_df: pd.DataFrame, region_level: str, region_key: str | None = None) -> pd.DataFrame:
    if universe_df.empty:
        return pd.DataFrame(columns=["aptSeq", "label"])
    frame = _ensure_region_name(universe_df, region_level)
    if region_key:
        frame = frame[frame["region_key"].astype(str) == str(region_key)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["aptSeq", "label"])
    label_col = frame.get("apt_name_repr", frame["aptSeq"])
    frame = frame.assign(label=label_col.fillna(frame["aptSeq"]).astype(str))
    return frame[["aptSeq", "label"]].drop_duplicates().sort_values("label").reset_index(drop=True)


def _prepare_model_frame(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    group_col: str | None = None,
    weight_col: str | None = None,
    log_target: bool = False,
    min_rows: int = 80,
) -> tuple[pd.DataFrame, list[str], pd.Series | None]:
    required = [target_col, *feature_cols]
    if group_col:
        required.append(group_col)
    if weight_col:
        required.append(weight_col)
    available = [column for column in required if column in df.columns]
    model_df = df[available].copy()
    if target_col not in model_df.columns:
        return pd.DataFrame(), [], None

    model_df[target_col] = _numeric(model_df[target_col])
    for column in feature_cols:
        if column in model_df.columns:
            values = model_df[column]
            if values.dtype == bool:
                model_df[column] = values.astype(int)
            else:
                model_df[column] = _numeric(values)
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    if log_target:
        model_df = model_df[model_df[target_col] > 0].copy()
        model_df["target_model"] = np.log(model_df[target_col])
    else:
        model_df = model_df[model_df[target_col].notna()].copy()
        model_df["target_model"] = model_df[target_col]
    if model_df.empty:
        return pd.DataFrame(), [], None

    usable_features = [column for column in feature_cols if column in model_df.columns and model_df[column].notna().any()]
    if not usable_features:
        return pd.DataFrame(), [], None
    for column in usable_features:
        model_df[column] = model_df[column].fillna(model_df[column].median())

    if group_col and group_col in model_df.columns:
        counts = model_df[group_col].value_counts(dropna=True)
        valid_groups = counts[counts > 1].index
        model_df = model_df[model_df[group_col].isin(valid_groups)].copy()
        if model_df.empty:
            return pd.DataFrame(), [], None
        demean_cols = ["target_model", *usable_features]
        group_means = model_df.groupby(group_col, observed=True)[demean_cols].transform("mean")
        model_df["target_model"] = model_df["target_model"] - group_means["target_model"]
        model_df[usable_features] = model_df[usable_features] - group_means[usable_features]

    std = model_df[usable_features].std(ddof=0).replace(0, np.nan)
    usable_features = [column for column in usable_features if pd.notna(std.get(column))]
    if not usable_features:
        return pd.DataFrame(), [], None
    means = model_df[usable_features].mean()
    std = model_df[usable_features].std(ddof=0).replace(0, np.nan)
    model_df[usable_features] = (model_df[usable_features] - means) / std
    model_df = model_df.dropna(subset=["target_model", *usable_features]).copy()
    if len(model_df) < min_rows:
        return pd.DataFrame(), [], None

    weights = None
    if weight_col and weight_col in model_df.columns:
        weights = _numeric(model_df[weight_col]).fillna(1.0).clip(lower=1.0)
    return model_df, usable_features, weights


def _fit_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    group_col: str | None = None,
    weight_col: str | None = None,
    log_target: bool = False,
    min_rows: int = 80,
) -> ModelResult:
    model_df, usable_features, weights = _prepare_model_frame(
        df,
        target_col,
        feature_cols,
        group_col=group_col,
        weight_col=weight_col,
        log_target=log_target,
        min_rows=min_rows,
    )
    if model_df.empty or not usable_features:
        return ModelResult(pd.DataFrame(), {"n_obs": 0.0, "r_squared": np.nan})

    X = sm.add_constant(model_df[usable_features], has_constant="add")
    y = model_df["target_model"]
    model = sm.WLS(y, X, weights=weights) if weights is not None else sm.OLS(y, X)
    result = model.fit(cov_type="HC3")
    params = result.params.drop("const", errors="ignore")
    conf = result.conf_int().rename(columns={0: "ci_low", 1: "ci_high"}).loc[params.index]
    coefficients = (
        pd.DataFrame(
            {
                "feature": params.index,
                "effect_value": params.to_numpy(),
                "ci_low": conf["ci_low"].to_numpy(),
                "ci_high": conf["ci_high"].to_numpy(),
                "pvalue": result.pvalues.drop("const", errors="ignore").reindex(params.index).to_numpy(),
                "tvalue": result.tvalues.drop("const", errors="ignore").reindex(params.index).to_numpy(),
            }
        )
        .assign(feature_label=lambda frame: frame["feature"].map(FEATURE_LABELS).fillna(frame["feature"]))
        .sort_values("effect_value", key=lambda series: series.abs(), ascending=False)
        .reset_index(drop=True)
    )
    return ModelResult(coefficients, {"n_obs": float(result.nobs), "r_squared": float(result.rsquared)})


def build_representative_coverage_frame(universe_df: pd.DataFrame, region_level: str = "sigungu") -> pd.DataFrame:
    if universe_df.empty:
        return pd.DataFrame()
    frame = _ensure_region_name(universe_df, region_level)
    grouped = (
        frame.groupby(["region_key", "region_label"], observed=True)
        .agg(
            complex_count=("aptSeq", "nunique"),
            any_59=("has_59_any", "sum"),
            any_84=("has_84_any", "sum"),
            pair_count=("is_pair_complex", "sum"),
            trade_pair_count=("is_trade_pair_complex", "sum"),
            rent_pair_count=("is_rent_pair_complex", "sum"),
        )
        .reset_index()
        .sort_values(["pair_count", "complex_count"], ascending=[False, False])
    )
    grouped["pair_share_pct"] = grouped["pair_count"] / grouped["complex_count"].replace(0, np.nan) * 100
    return grouped.reset_index(drop=True)


def build_representative_coverage_chart(coverage_df: pd.DataFrame) -> go.Figure:
    if coverage_df.empty:
        return _empty_figure("대표단지 coverage 데이터가 없습니다.")
    top = coverage_df.head(20).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["region_label"], y=top["complex_count"], name="대표단지 수", marker_color="#355C7D"))
    fig.add_trace(go.Bar(x=top["region_label"], y=top["pair_count"], name="Pair 단지 수", marker_color="#F67280"))
    fig.update_layout(title="지역별 대표단지 coverage", barmode="group", height=480, xaxis_title="", yaxis_title="단지 수")
    return fig


def build_region_timeline_frame(
    region_df: pd.DataFrame,
    *,
    region_level: str,
    market_type: str,
    area_band: int,
    region_key: str,
) -> pd.DataFrame:
    if region_df.empty:
        return pd.DataFrame()
    subset = region_df[
        (region_df["region_level"] == region_level)
        & (region_df["market_type"] == market_type)
        & (pd.to_numeric(region_df["area_band"], errors="coerce") == area_band)
        & (region_df["region_code"].astype(str) == str(region_key))
    ].copy()
    return _ensure_datetime(subset).sort_values("date").reset_index(drop=True)


def build_region_timeline_chart(trend_df: pd.DataFrame, title: str) -> go.Figure:
    if trend_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["complex_eq_median_py"],
            mode="lines",
            name="대표 평당가",
            line={"color": "#264653", "width": 2.5},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["p25"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["p75"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            name="IQR",
            fillcolor="rgba(38,70,83,0.15)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=trend_df["date"],
            y=trend_df["complex_count_observed"],
            name="관측 단지 수",
            marker_color="rgba(231,111,81,0.45)",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["complex_count_active"],
            mode="lines",
            name="활성 단지 수",
            line={"color": "#2A9D8F", "width": 2},
        ),
        secondary_y=True,
    )
    fig.update_layout(title=title, height=520, hovermode="x unified")
    fig.update_yaxes(title_text="만원/평", secondary_y=False)
    fig.update_yaxes(title_text="단지 수", secondary_y=True)
    return fig


def build_band_comparison_frame(
    region_df: pd.DataFrame,
    *,
    region_level: str,
    market_type: str,
    region_key: str,
) -> pd.DataFrame:
    if region_df.empty:
        return pd.DataFrame()
    subset = region_df[
        (region_df["region_level"] == region_level)
        & (region_df["market_type"] == market_type)
        & (region_df["region_code"].astype(str) == str(region_key))
    ].copy()
    if subset.empty:
        return subset
    pivot = (
        subset.pivot_table(
            index=["date", "ym", "region_code", "region_name"],
            columns="area_band",
            values=["complex_eq_median_py", "complex_count_active", "complex_count_observed"],
            observed=True,
        )
        .sort_index()
    )
    pivot.columns = [f"{metric}_{int(area_band)}" for metric, area_band in pivot.columns]
    result = pivot.reset_index()
    if {"complex_eq_median_py_59", "complex_eq_median_py_84"}.issubset(result.columns):
        result["gap_ratio"] = (result["complex_eq_median_py_84"] / result["complex_eq_median_py_59"] - 1) * 100
        result["gap_abs"] = result["complex_eq_median_py_84"] - result["complex_eq_median_py_59"]
    return result.sort_values("date").reset_index(drop=True)


def build_band_comparison_chart(comparison_df: pd.DataFrame, title: str) -> go.Figure:
    if comparison_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for band, color in [(59, "#457B9D"), (84, "#E76F51")]:
        column = f"complex_eq_median_py_{band}"
        if column in comparison_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=comparison_df["date"],
                    y=comparison_df[column],
                    mode="lines",
                    name=f"{AREA_BAND_LABELS[band]} 평당가",
                    line={"color": color, "width": 2.5},
                ),
                secondary_y=False,
            )
    if "gap_ratio" in comparison_df.columns:
        fig.add_trace(
            go.Scatter(
                x=comparison_df["date"],
                y=comparison_df["gap_ratio"],
                mode="lines",
                name="84/59 격차율",
                line={"color": "#2A9D8F", "width": 2, "dash": "dash"},
            ),
            secondary_y=True,
        )
    fig.update_layout(title=title, height=520, hovermode="x unified")
    fig.update_yaxes(title_text="만원/평", secondary_y=False)
    fig.update_yaxes(title_text="격차율(%)", secondary_y=True)
    return fig


def build_snapshot_distribution_frame(
    trade_band_df: pd.DataFrame,
    *,
    ym: str,
    region_level: str,
    area_band: int,
    top_n: int = 12,
) -> pd.DataFrame:
    if trade_band_df.empty:
        return pd.DataFrame()
    frame = _ensure_region_name(trade_band_df, region_level)
    frame = frame[
        (frame["ym"].astype(str) == str(ym))
        & (pd.to_numeric(frame["area_band"], errors="coerce") == area_band)
        & (pd.to_numeric(frame["price_per_py_filled"], errors="coerce").notna())
    ].copy()
    if frame.empty:
        return frame
    top_regions = (
        frame.groupby("region_label", observed=True)["aptSeq"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    return frame[frame["region_label"].isin(top_regions)].copy()


def build_snapshot_distribution_chart(distribution_df: pd.DataFrame, title: str) -> go.Figure:
    if distribution_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    fig = px.box(
        distribution_df,
        x="region_label",
        y="price_per_py_filled",
        points=False,
        color="region_label",
        title=title,
        labels={"region_label": "지역", "price_per_py_filled": "만원/평"},
    )
    fig.update_layout(height=520, showlegend=False)
    return fig


def build_pair_gap_history_frame(pair_gap_df: pd.DataFrame, apt_seq: str) -> pd.DataFrame:
    if pair_gap_df.empty:
        return pd.DataFrame()
    frame = pair_gap_df[pair_gap_df["aptSeq"].astype(str) == str(apt_seq)].copy()
    return _ensure_datetime(frame).sort_values("date").reset_index(drop=True)


def build_pair_gap_history_chart(gap_df: pd.DataFrame, title: str) -> go.Figure:
    if gap_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "sale_py_59_roll3" in gap_df.columns:
        fig.add_trace(
            go.Scatter(x=gap_df["date"], y=gap_df["sale_py_59_roll3"], mode="lines", name="59형", line={"color": "#457B9D"}),
            secondary_y=False,
        )
    if "sale_py_84_roll3" in gap_df.columns:
        fig.add_trace(
            go.Scatter(x=gap_df["date"], y=gap_df["sale_py_84_roll3"], mode="lines", name="84형", line={"color": "#E76F51"}),
            secondary_y=False,
        )
    if "sale_gap_ratio" in gap_df.columns:
        fig.add_trace(
            go.Scatter(
                x=gap_df["date"],
                y=gap_df["sale_gap_ratio"],
                mode="lines",
                name="84/59 격차율",
                line={"color": "#2A9D8F", "dash": "dash"},
            ),
            secondary_y=True,
        )
    fig.update_layout(title=title, height=520, hovermode="x unified")
    fig.update_yaxes(title_text="만원/평", secondary_y=False)
    fig.update_yaxes(title_text="격차율(%)", secondary_y=True)
    return fig


def build_region_spread_frame(
    region_df: pd.DataFrame,
    *,
    region_level: str,
    market_type: str,
) -> pd.DataFrame:
    if region_df.empty:
        return pd.DataFrame()
    subset = region_df[
        (region_df["region_level"] == region_level)
        & (region_df["market_type"] == market_type)
        & (pd.to_numeric(region_df["area_band"], errors="coerce").isin([59, 84]))
    ].copy()
    if subset.empty:
        return subset
    pivot = (
        subset.pivot_table(
            index=["region_code", "region_name", "ym", "date"],
            columns="area_band",
            values="complex_eq_median_py",
            observed=True,
        )
        .reset_index()
        .rename(columns={59: "price_59", 84: "price_84"})
        .sort_values(["region_code", "date"])
    )
    if {"price_59", "price_84"}.issubset(pivot.columns):
        pivot["spread_ratio"] = (pivot["price_84"] / pivot["price_59"] - 1) * 100
        pivot["spread_abs"] = pivot["price_84"] - pivot["price_59"]
    return pivot.reset_index(drop=True)


def build_spread_ranking_frame(spread_df: pd.DataFrame) -> pd.DataFrame:
    if spread_df.empty:
        return pd.DataFrame()
    latest_ym = spread_df["ym"].astype(str).max()
    latest = spread_df[spread_df["ym"].astype(str) == str(latest_ym)].copy()
    return latest.sort_values("spread_ratio", ascending=False).reset_index(drop=True)


def build_spread_chart(spread_df: pd.DataFrame, region_code: str, title: str) -> go.Figure:
    if spread_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    subset = spread_df[spread_df["region_code"].astype(str) == str(region_code)].copy()
    if subset.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["price_59"], mode="lines", name="59형", line={"color": "#457B9D"}), secondary_y=False)
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["price_84"], mode="lines", name="84형", line={"color": "#E76F51"}), secondary_y=False)
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["spread_ratio"], mode="lines", name="격차율", line={"color": "#2A9D8F", "dash": "dash"}), secondary_y=True)
    fig.update_layout(title=title, height=500, hovermode="x unified")
    fig.update_yaxes(title_text="만원/평", secondary_y=False)
    fig.update_yaxes(title_text="격차율(%)", secondary_y=True)
    return fig


def build_jeonse_ratio_band_frame(
    trade_band_df: pd.DataFrame,
    rent_band_df: pd.DataFrame,
    *,
    region_level: str,
    region_key: str,
) -> pd.DataFrame:
    if trade_band_df.empty or rent_band_df.empty:
        return pd.DataFrame()
    sale = _ensure_region_name(trade_band_df, region_level)
    sale = sale[sale["region_key"].astype(str) == str(region_key)].copy()
    sale = sale[["aptSeq", "ym", "date", "area_band", "price_per_py_filled"]]

    jeonse = rent_band_df[rent_band_df["rentType"] == "전세"].copy()
    jeonse = _ensure_region_name(jeonse, region_level)
    jeonse = jeonse[jeonse["region_key"].astype(str) == str(region_key)].copy()
    jeonse = jeonse[["aptSeq", "ym", "date", "area_band", "deposit_per_py_filled"]]

    merged = sale.merge(jeonse, on=["aptSeq", "ym", "date", "area_band"], how="inner")
    if merged.empty:
        return merged
    merged["jeonse_ratio_py"] = merged["deposit_per_py_filled"] / merged["price_per_py_filled"] * 100
    frame = (
        merged.groupby(["ym", "date", "area_band"], observed=True)["jeonse_ratio_py"]
        .median()
        .reset_index()
        .sort_values("date")
    )
    pivot = frame.pivot_table(index=["ym", "date"], columns="area_band", values="jeonse_ratio_py", observed=True).reset_index().rename(columns={59: "ratio_59", 84: "ratio_84"})
    if {"ratio_59", "ratio_84"}.issubset(pivot.columns):
        pivot["gap_ratio"] = pivot["ratio_84"] - pivot["ratio_59"]
    return pivot


def build_jeonse_ratio_band_chart(ratio_df: pd.DataFrame, title: str) -> go.Figure:
    if ratio_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for band, column, color in [(59, "ratio_59", "#457B9D"), (84, "ratio_84", "#E76F51")]:
        if column in ratio_df.columns:
            fig.add_trace(go.Scatter(x=ratio_df["date"], y=ratio_df[column], mode="lines", name=f"{AREA_BAND_LABELS[band]} 전세가율", line={"color": color}), secondary_y=False)
    if "gap_ratio" in ratio_df.columns:
        fig.add_trace(go.Scatter(x=ratio_df["date"], y=ratio_df["gap_ratio"], mode="lines", name="84-59 전세가율 차이", line={"color": "#2A9D8F", "dash": "dash"}), secondary_y=True)
    fig.update_layout(title=title, height=500, hovermode="x unified")
    fig.update_yaxes(title_text="전세가율(%)", secondary_y=False)
    fig.update_yaxes(title_text="차이(pp)", secondary_y=True)
    return fig


def build_liquidity_frame(
    trade_band_df: pd.DataFrame,
    *,
    region_level: str,
    area_band: int,
    region_key: str | None = None,
) -> pd.DataFrame:
    if trade_band_df.empty:
        return pd.DataFrame()
    frame = _ensure_region_name(trade_band_df, region_level)
    frame = frame[pd.to_numeric(frame["area_band"], errors="coerce") == area_band].copy()
    if region_key:
        frame = frame[frame["region_key"].astype(str) == str(region_key)].copy()
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(["ym", "date"], observed=True)
        .agg(
            active_complexes=("aptSeq", lambda s: int(s.nunique())),
            observed_complexes=("trade_count_obs", lambda s: int(pd.to_numeric(s, errors="coerce").gt(0).sum())),
            avg_fill_age=("months_since_trade_obs", lambda s: float(_numeric(s).mean()) if _numeric(s).notna().any() else np.nan),
            imputed_share=("is_trade_imputed", lambda s: float(pd.Series(s).fillna(False).mean() * 100)),
            transaction_count_obs=("trade_count_obs", lambda s: int(pd.to_numeric(s, errors="coerce").sum())),
        )
        .reset_index()
        .sort_values("date")
    )
    return grouped


def build_liquidity_chart(liquidity_df: pd.DataFrame, title: str) -> go.Figure:
    if liquidity_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=liquidity_df["date"], y=liquidity_df["observed_complexes"], name="관측 단지", marker_color="rgba(69,123,157,0.55)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=liquidity_df["date"], y=liquidity_df["active_complexes"], mode="lines", name="활성 단지", line={"color": "#264653", "width": 2.5}), secondary_y=False)
    fig.add_trace(go.Scatter(x=liquidity_df["date"], y=liquidity_df["imputed_share"], mode="lines", name="보간 비중", line={"color": "#E76F51", "dash": "dash"}), secondary_y=True)
    fig.update_layout(title=title, height=500, hovermode="x unified")
    fig.update_yaxes(title_text="단지 수", secondary_y=False)
    fig.update_yaxes(title_text="보간 비중(%)", secondary_y=True)
    return fig


def _prepare_gap_model_frame(forecast_df: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_datetime(forecast_df)
    if frame.empty:
        return frame
    if "household_count" in frame.columns:
        frame["log_households"] = np.log1p(_numeric(frame["household_count"]).clip(lower=0))
    if "date" in frame.columns:
        frame["year"] = frame["date"].dt.year
    if "completion_year" in frame.columns and "year" in frame.columns:
        frame["complex_age"] = frame["year"] - _numeric(frame["completion_year"])
        frame.loc[frame["complex_age"] < 0, "complex_age"] = np.nan
    if "usdkrw" in frame.columns:
        frame["usdkrw_change_3m"] = frame.groupby("aptSeq", observed=True)["usdkrw"].diff(3)
    frame["rate_x_parking"] = _numeric(frame.get("bok_rate_change_3m", pd.Series(index=frame.index))) * _numeric(frame.get("parking_per_household", pd.Series(index=frame.index)))
    frame["rate_x_density"] = _numeric(frame.get("bok_rate_change_3m", pd.Series(index=frame.index))) * (1.0 / _numeric(frame.get("floor_area_ratio", pd.Series(index=frame.index))).replace(0, np.nan))
    frame["liquidity_x_redevelop"] = _numeric(frame.get("m2_yoy", pd.Series(index=frame.index))) * _numeric(frame.get("redevelopment_option_score", pd.Series(index=frame.index)))
    return frame


def build_gap_rolling_coefficient_frame(forecast_df: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_gap_model_frame(forecast_df)
    if frame.empty or "sale_gap_ratio" not in frame.columns:
        return pd.DataFrame()
    yearly = (
        frame.groupby(["aptSeq", "year"], observed=True)
        .agg(
            sale_gap_ratio=("sale_gap_ratio", "mean"),
            log_households=("log_households", "last"),
            parking_per_household=("parking_per_household", "last"),
            floor_area_ratio=("floor_area_ratio", "last"),
            avg_land_area_per_household=("avg_land_area_per_household", "last"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for year, year_df in yearly.groupby("year", observed=True):
        result = _fit_regression(
            year_df,
            "sale_gap_ratio",
            ["log_households", "parking_per_household", "floor_area_ratio", "avg_land_area_per_household"],
            log_target=False,
            min_rows=100,
        )
        if result.coefficients.empty:
            continue
        for _, row in result.coefficients.iterrows():
            rows.append({"year": int(year), "feature": row["feature"], "feature_label": row["feature_label"], "coefficient": row["effect_value"]})
    return pd.DataFrame(rows)


def build_gap_rolling_coefficient_chart(rolling_df: pd.DataFrame) -> go.Figure:
    if rolling_df.empty:
        return _empty_figure("롤링 계수 데이터가 없습니다.")
    fig = px.line(
        rolling_df,
        x="year",
        y="coefficient",
        color="feature_label",
        markers=True,
        title="대표평형 gap의 시변 계수",
        labels={"year": "연도", "coefficient": "표준화 계수", "feature_label": "특성"},
    )
    fig.update_layout(height=500)
    return fig


def run_gap_panel_fixed_effects(forecast_df: pd.DataFrame) -> ModelResult:
    frame = _prepare_gap_model_frame(forecast_df)
    if frame.empty:
        return ModelResult(pd.DataFrame(), {"n_obs": 0.0, "r_squared": np.nan})
    target_col = "sale_gap_ratio_change_1m" if "sale_gap_ratio_change_1m" in frame.columns else "future_sale_gap_ratio_change_1m"
    return _fit_regression(
        frame,
        target_col,
        ["bok_rate_change_3m", "m2_yoy", "usdkrw_change_3m", "rate_x_parking", "rate_x_density", "liquidity_x_redevelop"],
        group_col="aptSeq",
        log_target=False,
        min_rows=120,
    )


def build_regime_response_frame(forecast_df: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_gap_model_frame(forecast_df)
    if frame.empty or "future_sale_gap_ratio_change_3m" not in frame.columns:
        return pd.DataFrame()
    frame = frame.dropna(subset=["future_sale_gap_ratio_change_3m", "bok_rate_change_3m", "m2_yoy"]).copy()
    if frame.empty:
        return frame
    m2_cut = frame["m2_yoy"].median()
    parking_cut = _numeric(frame.get("parking_per_household", pd.Series(index=frame.index))).quantile(0.75)
    density_cut = _numeric(frame.get("floor_area_ratio", pd.Series(index=frame.index))).quantile(0.25)
    scale_cut = _numeric(frame.get("household_count", pd.Series(index=frame.index))).quantile(0.75)
    frame["regime"] = np.where(frame["bok_rate_change_3m"] >= 0, "금리상승", "금리하락") + " / " + np.where(frame["m2_yoy"] >= m2_cut, "유동성확대", "유동성둔화")
    archetypes = {
        "대단지": _numeric(frame.get("household_count", pd.Series(index=frame.index))).ge(scale_cut),
        "주차 우수": _numeric(frame.get("parking_per_household", pd.Series(index=frame.index))).ge(parking_cut),
        "저밀도": _numeric(frame.get("floor_area_ratio", pd.Series(index=frame.index))).le(density_cut),
    }
    rows: list[dict[str, object]] = []
    for regime, regime_df in frame.groupby("regime", observed=True):
        for label, mask in archetypes.items():
            aligned = mask.loc[regime_df.index]
            strong = regime_df.loc[aligned, "future_sale_gap_ratio_change_3m"].mean()
            base = regime_df.loc[~aligned, "future_sale_gap_ratio_change_3m"].mean()
            if pd.isna(strong) or pd.isna(base):
                continue
            rows.append({"regime": regime, "archetype": label, "premium_pp": strong - base})
    return pd.DataFrame(rows)


def build_regime_response_chart(regime_df: pd.DataFrame) -> go.Figure:
    if regime_df.empty:
        return _empty_figure("거시 국면별 반응 데이터가 없습니다.")
    fig = px.bar(
        regime_df,
        x="regime",
        y="premium_pp",
        color="archetype",
        barmode="group",
        title="거시 국면별 84/59 gap 반응",
        labels={"premium_pp": "추가 변화(pp)", "regime": "거시 국면", "archetype": "유형"},
    )
    fig.update_layout(height=480)
    return fig


def build_spillover_frame(region_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    spread_df = build_region_spread_frame(region_df, region_level="bjdong", market_type="sale")
    if spread_df.empty:
        return pd.DataFrame(), {"lag1_corr": np.nan, "spread_pp": np.nan}
    latest = spread_df.groupby("region_code", observed=True)["price_84"].last()
    leader_cut = latest.quantile(0.75)
    spread_df["leader_flag"] = spread_df["region_code"].map(latest).ge(leader_cut)
    monthly = (
        spread_df.groupby(["date", "leader_flag"], observed=True)["spread_ratio"]
        .mean()
        .unstack("leader_flag")
        .rename(columns={False: "follower_spread", True: "leader_spread"})
        .dropna()
        .reset_index()
        .sort_values("date")
    )
    if monthly.empty:
        return pd.DataFrame(), {"lag1_corr": np.nan, "spread_pp": np.nan}
    monthly["leader_spread_lag1"] = monthly["leader_spread"].shift(1)
    corr_df = monthly.dropna(subset=["leader_spread_lag1", "follower_spread"])
    lag1_corr = float(corr_df["leader_spread_lag1"].corr(corr_df["follower_spread"])) if not corr_df.empty else np.nan
    spread = float((monthly["leader_spread"] - monthly["follower_spread"]).mean())
    return monthly, {"lag1_corr": lag1_corr, "spread_pp": spread}


def build_spillover_chart(spillover_df: pd.DataFrame) -> go.Figure:
    if spillover_df.empty:
        return _empty_figure("확산 분석 데이터가 없습니다.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spillover_df["date"], y=spillover_df["leader_spread"], mode="lines", name="선도 지역", line={"color": "#264653"}))
    fig.add_trace(go.Scatter(x=spillover_df["date"], y=spillover_df["follower_spread"], mode="lines", name="추종 지역", line={"color": "#E76F51"}))
    fig.update_layout(title="지역 간 spread 확산", height=460, yaxis_title="84/59 격차율(%)", hovermode="x unified")
    return fig


def build_mean_reversion_frame(forecast_df: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_gap_model_frame(forecast_df)
    required = ["sale_gap_ratio", "future_sale_gap_ratio_change_3m", "future_sale_gap_ratio_change_12m"]
    if frame.empty or not set(required).issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.dropna(subset=required).copy()
    if frame.empty:
        return frame
    frame["gap_bucket"] = _safe_qcut(frame["sale_gap_ratio"], 5, ["하위", "중하", "중위", "중상", "상위"])
    grouped = (
        frame.groupby("gap_bucket", observed=True)
        .agg(
            current_gap=("sale_gap_ratio", "median"),
            mean_change_3m=("future_sale_gap_ratio_change_3m", "mean"),
            mean_change_12m=("future_sale_gap_ratio_change_12m", "mean"),
            count=("aptSeq", "size"),
        )
        .reset_index()
    )
    return grouped


def build_mean_reversion_chart(reversion_df: pd.DataFrame) -> go.Figure:
    if reversion_df.empty:
        return _empty_figure("평균회귀 데이터가 없습니다.")
    melted = reversion_df.melt(
        id_vars=["gap_bucket", "count"],
        value_vars=["mean_change_3m", "mean_change_12m"],
        var_name="horizon",
        value_name="delta_pp",
    )
    melted["horizon"] = melted["horizon"].map({"mean_change_3m": "3개월 후", "mean_change_12m": "12개월 후"})
    fig = px.bar(
        melted,
        x="gap_bucket",
        y="delta_pp",
        color="horizon",
        barmode="group",
        title="현재 gap 수준별 향후 변화",
        labels={"gap_bucket": "현재 gap 분위", "delta_pp": "향후 변화(pp)", "horizon": "시차"},
    )
    fig.update_layout(height=440)
    return fig


def _forecast_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in FORECAST_FEATURES if column in df.columns]


def _run_forecast_model(
    forecast_df: pd.DataFrame,
    target_col: str,
    *,
    test_periods: int = 6,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    frame = _prepare_gap_model_frame(forecast_df)
    if frame.empty or target_col not in frame.columns:
        return pd.DataFrame(), {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_test": 0.0}, pd.DataFrame()

    feature_cols = _forecast_feature_columns(frame)
    available = [column for column in ["date", "ym", target_col, *feature_cols] if column in frame.columns]
    model_df = frame[available].copy()
    model_df[target_col] = _numeric(model_df[target_col])
    for column in feature_cols:
        values = model_df[column]
        model_df[column] = values.astype(int) if values.dtype == bool else _numeric(values)
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).copy()
    if model_df.empty:
        return pd.DataFrame(), {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_test": 0.0}, pd.DataFrame()

    ym_values = sorted(model_df["ym"].dropna().astype(str).unique())
    if len(ym_values) < 10:
        return pd.DataFrame(), {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_test": 0.0}, pd.DataFrame()
    test_periods = min(test_periods, max(3, len(ym_values) // 4))
    train_ym = ym_values[:-test_periods]
    test_ym = ym_values[-test_periods:]
    train_df = model_df[model_df["ym"].isin(train_ym)].copy()
    test_df = model_df[model_df["ym"].isin(test_ym)].copy()
    usable_features = [column for column in feature_cols if column in train_df.columns and train_df[column].notna().any()]
    if not usable_features or train_df.empty or test_df.empty:
        return pd.DataFrame(), {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_test": 0.0}, pd.DataFrame()

    medians = train_df[usable_features].median()
    train_X = train_df[usable_features].fillna(medians).astype(float)
    test_X = test_df[usable_features].fillna(medians).astype(float)
    means = train_X.mean()
    std = train_X.std(ddof=0).replace(0, np.nan)
    usable_features = [column for column in usable_features if pd.notna(std.get(column))]
    if not usable_features:
        return pd.DataFrame(), {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_test": 0.0}, pd.DataFrame()
    train_X = (train_X[usable_features] - means[usable_features]) / std[usable_features]
    test_X = (test_X[usable_features] - means[usable_features]) / std[usable_features]
    y_train = train_df[target_col].astype(float)
    model = sm.OLS(y_train, sm.add_constant(train_X, has_constant="add")).fit(cov_type="HC3")
    predictions = model.predict(sm.add_constant(test_X, has_constant="add"))

    pred_df = test_df[["date", "ym", target_col]].copy()
    pred_df["prediction"] = predictions
    pred_df = (
        pred_df.groupby(["date", "ym"], observed=True)
        .agg(actual=(target_col, "mean"), prediction=("prediction", "mean"))
        .reset_index()
        .sort_values("date")
    )
    residual = pred_df["actual"] - pred_df["prediction"]
    mae = float(np.abs(residual).mean())
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mape_base = pred_df["actual"].abs().replace(0, np.nan)
    mape = float((np.abs(residual) / mape_base).dropna().mean() * 100) if mape_base.notna().any() else np.nan
    directional_accuracy = float((np.sign(pred_df["actual"]) == np.sign(pred_df["prediction"])).mean() * 100) if not pred_df.empty else np.nan
    params = model.params.drop("const", errors="ignore")
    importance = (
        pd.DataFrame({"feature": params.index, "coefficient": params.to_numpy()})
        .assign(feature_label=lambda frame: frame["feature"].map(FEATURE_LABELS).fillna(frame["feature"]))
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    metrics = {"mae": mae, "rmse": rmse, "mape": mape, "directional_accuracy": directional_accuracy, "n_test": float(len(test_df))}
    return pred_df, metrics, importance


def run_sale_band_forecast(forecast_df: pd.DataFrame, band: int, horizon: int) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    return _run_forecast_model(forecast_df, f"future_sale_py_{band}_return_{horizon}m", test_periods=6 if horizon < 12 else 12)


def run_gap_forecast(forecast_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    return _run_forecast_model(forecast_df, f"future_sale_gap_ratio_change_{horizon}m", test_periods=6 if horizon < 12 else 12)


def build_screening_frame(forecast_df: pd.DataFrame, horizon: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _prepare_gap_model_frame(forecast_df)
    target_specs = {
        "pred_59_return": f"future_sale_py_59_return_{horizon}m",
        "pred_84_return": f"future_sale_py_84_return_{horizon}m",
        "pred_gap_change": f"future_sale_gap_ratio_change_{horizon}m",
    }
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    latest_ym = frame["ym"].astype(str).max()
    latest_df = frame[frame["ym"].astype(str) == str(latest_ym)].copy()
    if latest_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    feature_cols = _forecast_feature_columns(frame)
    for output_col, target_col in target_specs.items():
        labeled = frame.dropna(subset=[target_col]).copy()
        if labeled.empty or target_col not in labeled.columns:
            continue
        usable_features = [column for column in feature_cols if column in labeled.columns and labeled[column].notna().any()]
        if not usable_features:
            continue
        for column in usable_features:
            values = labeled[column]
            labeled[column] = values.astype(int) if values.dtype == bool else _numeric(values)
            latest_df[column] = latest_df[column].astype(int) if latest_df[column].dtype == bool else _numeric(latest_df[column])
        medians = labeled[usable_features].median()
        train_X = labeled[usable_features].fillna(medians).astype(float)
        score_X = latest_df[usable_features].fillna(medians).astype(float)
        means = train_X.mean()
        std = train_X.std(ddof=0).replace(0, np.nan)
        usable_features = [column for column in usable_features if pd.notna(std.get(column))]
        if not usable_features:
            continue
        train_X = (train_X[usable_features] - means[usable_features]) / std[usable_features]
        score_X = (score_X[usable_features] - means[usable_features]) / std[usable_features]
        y_train = _numeric(labeled[target_col]).astype(float)
        model = sm.OLS(y_train, sm.add_constant(train_X, has_constant="add")).fit(cov_type="HC3")
        latest_df[output_col] = model.predict(sm.add_constant(score_X, has_constant="add"))

    complex_rank = latest_df[
        [
            "aptSeq",
            "apt_name_repr",
            "sigungu_name",
            "dong_repr",
            "pred_59_return",
            "pred_84_return",
            "pred_gap_change",
        ]
    ].dropna(how="all", subset=["pred_59_return", "pred_84_return", "pred_gap_change"])
    region_rank = (
        complex_rank.groupby(["sigungu_name", "dong_repr"], observed=True)[["pred_59_return", "pred_84_return", "pred_gap_change"]]
        .mean()
        .reset_index()
        .sort_values("pred_gap_change", ascending=False)
    )
    return region_rank.reset_index(drop=True), complex_rank.sort_values("pred_gap_change", ascending=False).reset_index(drop=True)


def build_screening_chart(region_rank_df: pd.DataFrame) -> go.Figure:
    if region_rank_df.empty:
        return _empty_figure("스크리닝 데이터가 없습니다.")
    top = pd.concat([region_rank_df.head(10), region_rank_df.tail(10)]).drop_duplicates().copy()
    fig = px.bar(
        top,
        x="sigungu_name",
        y="pred_gap_change",
        color="pred_gap_change",
        color_continuous_scale="RdBu",
        title="향후 spread 확대/축소 지역 스크리닝",
        labels={"sigungu_name": "지역", "pred_gap_change": "예상 gap 변화(pp)"},
    )
    fig.update_layout(height=440, coloraxis_showscale=False)
    return fig


def build_scenario_frame(
    forecast_df: pd.DataFrame,
    *,
    rate_delta: float,
    liquidity_delta: float,
    fx_delta: float,
) -> pd.DataFrame:
    frame = _prepare_gap_model_frame(forecast_df)
    target_col = "future_sale_gap_ratio_change_12m" if "future_sale_gap_ratio_change_12m" in frame.columns else "future_sale_gap_ratio_change_3m"
    if frame.empty or target_col not in frame.columns:
        return pd.DataFrame()
    features = [column for column in ["log_households", "parking_per_household", "floor_area_ratio", "avg_land_area_per_household", "redevelopment_option_score", "bok_rate_change_3m", "m2_yoy", "usdkrw_change_3m"] if column in frame.columns]
    train = frame.dropna(subset=[target_col]).copy()
    if len(train) < 120 or len(features) < 4:
        return pd.DataFrame()
    for column in features:
        train[column] = _numeric(train[column]).fillna(_numeric(train[column]).median())
    model = sm.OLS(train[target_col], sm.add_constant(train[features], has_constant="add")).fit(cov_type="HC3")
    quantiles = train[features].quantile([0.1, 0.5, 0.9])
    medians = train[features].median()
    archetypes = {
        "기준": medians.to_dict(),
        "대단지": {**medians.to_dict(), "log_households": float(quantiles.loc[0.9, "log_households"]) if "log_households" in quantiles.columns else float(medians.get("log_households", 0.0))},
        "주차 우수": {**medians.to_dict(), "parking_per_household": float(quantiles.loc[0.9, "parking_per_household"]) if "parking_per_household" in quantiles.columns else float(medians.get("parking_per_household", 0.0))},
        "저밀도": {**medians.to_dict(), "floor_area_ratio": float(quantiles.loc[0.1, "floor_area_ratio"]) if "floor_area_ratio" in quantiles.columns else float(medians.get("floor_area_ratio", 0.0))},
        "재건축 잠재력": {**medians.to_dict(), "redevelopment_option_score": float(quantiles.loc[0.9, "redevelopment_option_score"]) if "redevelopment_option_score" in quantiles.columns else float(medians.get("redevelopment_option_score", 0.0))},
    }
    rows: list[dict[str, object]] = []
    for label, feature_row in archetypes.items():
        scenario_row = feature_row.copy()
        if "bok_rate_change_3m" in scenario_row:
            scenario_row["bok_rate_change_3m"] = float(scenario_row["bok_rate_change_3m"]) + rate_delta
        if "m2_yoy" in scenario_row:
            scenario_row["m2_yoy"] = float(scenario_row["m2_yoy"]) + liquidity_delta
        if "usdkrw_change_3m" in scenario_row:
            scenario_row["usdkrw_change_3m"] = float(scenario_row["usdkrw_change_3m"]) + fx_delta
        base_pred = float(model.predict(sm.add_constant(pd.DataFrame([feature_row])[features], has_constant="add")).iloc[0])
        scenario_pred = float(model.predict(sm.add_constant(pd.DataFrame([scenario_row])[features], has_constant="add")).iloc[0])
        rows.append({"archetype": label, "base_change": base_pred, "scenario_change": scenario_pred, "incremental_change": scenario_pred - base_pred})
    return pd.DataFrame(rows)


def build_scenario_chart(scenario_df: pd.DataFrame) -> go.Figure:
    if scenario_df.empty:
        return _empty_figure("시나리오 데이터가 없습니다.")
    fig = go.Figure(go.Bar(x=scenario_df["archetype"], y=scenario_df["incremental_change"], marker_color=np.where(scenario_df["incremental_change"] >= 0, "#2A9D8F", "#E76F51")))
    fig.update_layout(title="시나리오별 archetype 반응", height=420, yaxis_title="추가 gap 변화(pp)")
    return fig
