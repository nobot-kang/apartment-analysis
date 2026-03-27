"""Apartment-complex level analytics for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots

STATIC_COMPLEX_COLUMNS = [
    "aptSeq",
    "apt_name",
    "apt_name_repr",
    "dong_name",
    "dong_repr",
    "sigungu_code",
    "bjdong_code",
    "completion_year",
    "household_count",
    "total_parking_count",
    "parking_per_household",
    "floor_area_ratio",
    "building_coverage_ratio",
    "avg_land_area_per_household",
    "avg_total_area_per_household",
    "ground_floor_count",
    "underground_floor_count",
    "complex_scale_bucket",
    "density_bucket",
    "redevelopment_option_score",
    "feature_missing_count",
    "floor_area_ratio_missing",
    "building_coverage_ratio_missing",
    "avg_land_area_per_household_missing",
    "avg_total_area_per_household_missing",
    "total_parking_count_missing",
    "parking_per_household_missing",
]

SNAPSHOT_METRICS = {
    "trade_count": "sum",
    "trade_occurrence": "mean",
    "turnover_rate": "mean",
    "trade_price_mean": "mean",
    "trade_price_std84": "mean",
    "trade_price_per_m2": "mean",
    "jeonse_count": "sum",
    "jeonse_deposit_mean": "mean",
    "jeonse_deposit_std84": "mean",
    "jeonse_deposit_per_m2": "mean",
    "wolse_count": "sum",
    "wolse_deposit_mean": "mean",
    "wolse_monthly_rent_mean": "mean",
    "wolse_monthly_rent_per_m2": "mean",
    "jeonse_ratio": "mean",
    "conversion_rate": "mean",
    "trade_price_per_m2_yoy": "mean",
    "jeonse_deposit_per_m2_yoy": "mean",
    "wolse_monthly_rent_per_m2_yoy": "mean",
    "jeonse_ratio_yoy": "mean",
    "conversion_rate_yoy": "mean",
    "bok_rate": "mean",
    "bok_rate_change_3m": "mean",
    "m2": "mean",
    "m2_yoy": "mean",
    "usdkrw": "mean",
}

HEDONIC_FEATURES = [
    "log_households",
    "parking_per_household",
    "floor_area_ratio",
    "building_coverage_ratio",
    "avg_land_area_per_household",
    "ground_floor_count",
    "underground_floor_count",
    "complex_age",
    "redevelopment_option_score",
]

MISSING_FEATURES = [
    "parking_per_household_missing",
    "floor_area_ratio_missing",
    "building_coverage_ratio_missing",
    "avg_land_area_per_household_missing",
]

FEATURE_LABELS = {
    "log_households": "세대수",
    "parking_per_household": "세대당 주차대수",
    "floor_area_ratio": "용적률",
    "building_coverage_ratio": "건폐율",
    "avg_land_area_per_household": "세대당 대지면적",
    "ground_floor_count": "지상층수",
    "underground_floor_count": "지하층수",
    "complex_age": "준공연차",
    "redevelopment_option_score": "재건축 옵션 점수",
    "parking_per_household_missing": "주차정보 결측",
    "floor_area_ratio_missing": "용적률 결측",
    "building_coverage_ratio_missing": "건폐율 결측",
    "avg_land_area_per_household_missing": "대지지분 결측",
    "bok_rate_change_3m": "기준금리 3개월 변화",
    "m2_yoy": "M2 YoY",
    "rate_x_parking": "금리 x 주차",
    "rate_x_land": "금리 x 대지지분",
    "m2_x_scale": "유동성 x 세대수",
    "m2_x_redevelop": "유동성 x 재건축 옵션",
}

ARCHETYPE_ORDER = ["기준", "대단지", "주차 우수", "저밀도", "재건축 잠재력"]
ROLLING_FEATURES = ["log_households", "parking_per_household", "avg_land_area_per_household", "floor_area_ratio"]


@dataclass
class ModelResult:
    coefficients: pd.DataFrame
    metrics: dict[str, float]


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=480)
    return fig


def _existing_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result


def _safe_qcut(series: pd.Series, q: int, labels: Sequence[str]) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    if valid.notna().sum() < q:
        return pd.Series(pd.NA, index=series.index, dtype="string")
    ranked = valid.rank(method="first")
    try:
        buckets = pd.qcut(ranked, q=q, labels=labels)
        return pd.Series(buckets, index=series.index, dtype="string")
    except Exception:
        return pd.Series(pd.NA, index=series.index, dtype="string")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_coverage_report(master_df: pd.DataFrame) -> pd.DataFrame:
    """Return feature coverage diagnostics for the complex master table."""
    if master_df.empty:
        return pd.DataFrame(columns=["feature", "available_complexes", "coverage_pct", "median"])

    report_rows: list[dict[str, float | str]] = []
    feature_map = {
        "household_count": "세대수",
        "total_parking_count": "총주차수",
        "parking_per_household": "세대당 주차대수",
        "floor_area_ratio": "용적률",
        "building_coverage_ratio": "건폐율",
        "avg_land_area_per_household": "세대당 대지면적",
        "avg_total_area_per_household": "세대당 연면적",
        "redevelopment_option_score": "재건축 옵션 점수",
    }
    total = len(master_df)
    for column, label in feature_map.items():
        if column not in master_df.columns:
            continue
        values = _numeric(master_df[column])
        if column == "redevelopment_option_score":
            valid_values = values.notna()
        else:
            valid_values = values.gt(0)
        available = int(valid_values.sum())
        report_rows.append(
            {
                "feature": label,
                "available_complexes": available,
                "coverage_pct": available / total * 100,
                "median": float(values[valid_values].median()) if available else np.nan,
            }
        )
    return pd.DataFrame(report_rows).sort_values("coverage_pct", ascending=False).reset_index(drop=True)


def build_latest_snapshot(panel_df: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Aggregate a complex-month panel into a recent complex snapshot."""
    if panel_df.empty:
        return pd.DataFrame()

    panel = _ensure_datetime(panel_df)
    latest_date = panel["date"].max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    cutoff = latest_date - pd.DateOffset(months=max(months - 1, 0))
    subset = panel[panel["date"] >= cutoff].copy()
    if subset.empty:
        return pd.DataFrame()

    metric_agg = {column: agg for column, agg in SNAPSHOT_METRICS.items() if column in subset.columns}
    metrics = subset.groupby("aptSeq", observed=True).agg(metric_agg).reset_index()
    metrics = metrics.rename(
        columns={
            "trade_count": "trade_count_recent",
            "trade_occurrence": "trade_occurrence_rate",
            "turnover_rate": "turnover_rate_recent",
            "jeonse_count": "jeonse_count_recent",
            "wolse_count": "wolse_count_recent",
        }
    )
    metrics["observed_months"] = subset.groupby("aptSeq", observed=True)["ym"].nunique().reindex(metrics["aptSeq"]).to_numpy()
    metrics["latest_date"] = latest_date

    static_columns = [column for column in _existing_columns(subset, STATIC_COMPLEX_COLUMNS) if column != "aptSeq"]
    static_df = (
        subset.sort_values(["aptSeq", "date"])
        .drop_duplicates(subset=["aptSeq"], keep="last")[["aptSeq", *static_columns]]
        .copy()
    )
    snapshot = static_df.merge(metrics, on="aptSeq", how="outer")

    if "completion_year" in snapshot.columns and "latest_date" in snapshot.columns:
        snapshot["complex_age"] = snapshot["latest_date"].dt.year - _numeric(snapshot["completion_year"])
        snapshot.loc[snapshot["complex_age"] < 0, "complex_age"] = np.nan
    if "household_count" in snapshot.columns:
        snapshot["log_households"] = np.log1p(_numeric(snapshot["household_count"]).clip(lower=0))
    snapshot["price_segment"] = _safe_qcut(snapshot.get("trade_price_per_m2", pd.Series(index=snapshot.index, dtype="float64")), 3, ["저가", "중가", "고가"])
    snapshot["parking_bucket"] = _safe_qcut(snapshot.get("parking_per_household", pd.Series(index=snapshot.index, dtype="float64")), 4, ["하위", "중하", "중상", "상위"])
    return snapshot.sort_values(["dong_repr", "apt_name_repr"], na_position="last").reset_index(drop=True)


def build_yearly_snapshot(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a complex-month panel into a complex-year panel."""
    if panel_df.empty:
        return pd.DataFrame()

    panel = _ensure_datetime(panel_df)
    panel = panel.dropna(subset=["date"]).copy()
    panel["year"] = panel["date"].dt.year
    metric_agg = {column: agg for column, agg in SNAPSHOT_METRICS.items() if column in panel.columns}
    metrics = panel.groupby(["aptSeq", "year"], observed=True).agg(metric_agg).reset_index()
    if "trade_count" in metrics.columns:
        metrics = metrics.rename(columns={"trade_count": "annual_trade_count"})
    if "trade_occurrence" in metrics.columns:
        metrics = metrics.rename(columns={"trade_occurrence": "trade_occurrence_rate"})
    if "turnover_rate" in metrics.columns:
        metrics = metrics.rename(columns={"turnover_rate": "turnover_rate_mean"})

    static_columns = [column for column in _existing_columns(panel, STATIC_COMPLEX_COLUMNS) if column != "aptSeq"]
    static_df = (
        panel.sort_values(["aptSeq", "date"])
        .drop_duplicates(subset=["aptSeq"], keep="last")[["aptSeq", *static_columns]]
        .copy()
    )
    yearly = metrics.merge(static_df, on="aptSeq", how="left")
    yearly["complex_age"] = yearly["year"] - _numeric(yearly.get("completion_year", pd.Series(index=yearly.index)))
    yearly.loc[yearly["complex_age"] < 0, "complex_age"] = np.nan
    yearly["log_households"] = np.log1p(_numeric(yearly.get("household_count", pd.Series(index=yearly.index))).clip(lower=0))
    yearly["price_segment"] = _safe_qcut(yearly.get("trade_price_per_m2", pd.Series(index=yearly.index)), 3, ["저가", "중가", "고가"])
    return yearly.sort_values(["year", "aptSeq"]).reset_index(drop=True)


def build_complex_profile_frame(master_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise core complex attributes by dong."""
    if master_df.empty:
        return pd.DataFrame()

    frame = master_df.copy()
    for column in [
        "household_count",
        "parking_per_household",
        "floor_area_ratio",
        "building_coverage_ratio",
        "avg_land_area_per_household",
    ]:
        if column in frame.columns:
            frame[column] = _numeric(frame[column])

    grouped = (
        frame.groupby("dong_repr", observed=True)
        .agg(
            complex_count=("aptSeq", "nunique"),
            median_households=("household_count", "median"),
            median_parking=("parking_per_household", "median"),
            median_far=("floor_area_ratio", "median"),
            median_bcr=("building_coverage_ratio", "median"),
            median_land_share=("avg_land_area_per_household", "median"),
        )
        .reset_index()
        .sort_values(["complex_count", "median_households"], ascending=[False, False])
    )
    return grouped.reset_index(drop=True)


def build_complex_profile_chart(profile_df: pd.DataFrame) -> go.Figure:
    if profile_df.empty:
        return _empty_figure("법정동별 단지 특성 프로파일 데이터가 없습니다.")

    top = profile_df.head(20).copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=top["dong_repr"], y=top["complex_count"], name="단지 수", marker_color="#355C7D"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=top["dong_repr"],
            y=top["median_households"],
            mode="lines+markers",
            name="중위 세대수",
            line={"color": "#F67280", "width": 2.5},
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="법정동")
    fig.update_yaxes(title_text="단지 수", secondary_y=False)
    fig.update_yaxes(title_text="중위 세대수", secondary_y=True)
    fig.update_layout(title="법정동별 단지 프로파일", height=500, hovermode="x unified")
    return fig


def build_scale_premium_frame(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty or "complex_scale_bucket" not in snapshot_df.columns:
        return pd.DataFrame()

    grouped = (
        snapshot_df.groupby("complex_scale_bucket", observed=True)
        .agg(
            complex_count=("aptSeq", "nunique"),
            trade_price_per_m2=("trade_price_per_m2", "median"),
            jeonse_deposit_per_m2=("jeonse_deposit_per_m2", "median"),
            wolse_monthly_rent_per_m2=("wolse_monthly_rent_per_m2", "median"),
            jeonse_ratio=("jeonse_ratio", "median"),
        )
        .reset_index()
    )
    return grouped


def build_scale_premium_chart(scale_df: pd.DataFrame) -> go.Figure:
    if scale_df.empty:
        return _empty_figure("세대수 프리미엄 데이터가 없습니다.")

    melted = scale_df.melt(
        id_vars=["complex_scale_bucket", "complex_count"],
        value_vars=["trade_price_per_m2", "jeonse_deposit_per_m2", "wolse_monthly_rent_per_m2"],
        var_name="metric",
        value_name="value",
    )
    label_map = {
        "trade_price_per_m2": "매매 가격",
        "jeonse_deposit_per_m2": "전세 가격",
        "wolse_monthly_rent_per_m2": "월세 가격",
    }
    melted["metric"] = melted["metric"].map(label_map)
    fig = px.bar(
        melted,
        x="complex_scale_bucket",
        y="value",
        color="metric",
        barmode="group",
        title="대단지 프리미엄",
        labels={"complex_scale_bucket": "세대수 구간", "value": "중위 m2당 가격"},
    )
    fig.update_layout(height=480)
    return fig


def build_parking_premium_frame(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty or "parking_per_household" not in snapshot_df.columns:
        return pd.DataFrame()

    valid = snapshot_df[_numeric(snapshot_df["parking_per_household"]).gt(0)].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["parking_quantile"] = _safe_qcut(valid["parking_per_household"], 4, ["하위", "중하", "중상", "상위"])
    grouped = (
        valid.groupby("parking_quantile", observed=True)
        .agg(
            complex_count=("aptSeq", "nunique"),
            trade_price_per_m2=("trade_price_per_m2", "median"),
            jeonse_ratio=("jeonse_ratio", "median"),
            turnover_rate_recent=("turnover_rate_recent", "median"),
        )
        .reset_index()
    )
    return grouped


def build_parking_premium_chart(parking_df: pd.DataFrame) -> go.Figure:
    if parking_df.empty:
        return _empty_figure("주차 프리미엄을 계산할 데이터가 없습니다.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=parking_df["parking_quantile"],
            y=parking_df["trade_price_per_m2"],
            name="매매 가격",
            marker_color="#2A9D8F",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=parking_df["parking_quantile"],
            y=parking_df["jeonse_ratio"],
            name="전세가율",
            mode="lines+markers",
            line={"color": "#E76F51", "width": 2.5},
        ),
        secondary_y=True,
    )
    fig.update_layout(title="주차 프리미엄", height=480, hovermode="x unified")
    fig.update_yaxes(title_text="중위 m2당 매매가", secondary_y=False)
    fig.update_yaxes(title_text="전세가율(%)", secondary_y=True)
    return fig


def build_density_matrix(snapshot_df: pd.DataFrame, value_col: str = "trade_price_per_m2") -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame()
    required = {"floor_area_ratio", "building_coverage_ratio", value_col}
    if not required.issubset(snapshot_df.columns):
        return pd.DataFrame()

    frame = snapshot_df.copy()
    frame = frame.dropna(subset=list(required)).copy()
    if frame.empty:
        return pd.DataFrame()

    frame["far_band"] = pd.cut(frame["floor_area_ratio"], bins=[0, 200, 300, 400, np.inf], labels=["~200", "200~300", "300~400", "400+"], right=False)
    frame["bcr_band"] = pd.cut(frame["building_coverage_ratio"], bins=[0, 15, 25, 35, np.inf], labels=["~15", "15~25", "25~35", "35+"], right=False)
    matrix = (
        frame.groupby(["far_band", "bcr_band"], observed=True)[value_col]
        .median()
        .reset_index()
        .pivot(index="far_band", columns="bcr_band", values=value_col)
    )
    return matrix


def build_density_heatmap(matrix_df: pd.DataFrame, title: str) -> go.Figure:
    if matrix_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")

    fig = px.imshow(
        matrix_df,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={"x": "건폐율 구간", "y": "용적률 구간", "color": "가격"},
        title=title,
    )
    fig.update_layout(height=500)
    return fig


def build_land_premium_chart(snapshot_df: pd.DataFrame) -> go.Figure:
    if snapshot_df.empty:
        return _empty_figure("대지지분 프리미엄 데이터가 없습니다.")

    frame = snapshot_df.dropna(subset=["avg_land_area_per_household", "trade_price_per_m2"]).copy()
    if frame.empty:
        return _empty_figure("대지지분 프리미엄 데이터가 없습니다.")
    frame["household_bucket"] = _safe_qcut(frame["household_count"], 4, ["소형", "중형", "대단지", "초대형"])
    fig = px.scatter(
        frame,
        x="avg_land_area_per_household",
        y="trade_price_per_m2",
        color="household_bucket",
        size="household_count",
        hover_name="apt_name_repr",
        trendline="ols",
        title="세대당 평균대지면적과 매매 가격",
        labels={
            "avg_land_area_per_household": "세대당 평균대지면적",
            "trade_price_per_m2": "m2당 매매가",
            "household_bucket": "세대수 구간",
        },
    )
    fig.update_layout(height=520)
    return fig


def _prepare_regression_frame(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    group_col: str | None = None,
    weight_col: str | None = None,
    log_target: bool = True,
) -> tuple[pd.DataFrame, list[str], pd.Series | None]:
    required = [target_col, *feature_cols]
    if group_col:
        required.append(group_col)
    if weight_col:
        required.append(weight_col)
    model_df = df[_existing_columns(df, required)].copy()
    if target_col not in model_df.columns:
        return pd.DataFrame(), [], None

    model_df[target_col] = _numeric(model_df[target_col])
    for column in feature_cols:
        if column in model_df.columns:
            model_df[column] = _numeric(model_df[column])

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    if log_target:
        model_df = model_df[model_df[target_col] > 0].copy()
        model_df["target_model"] = np.log(model_df[target_col])
    else:
        model_df = model_df[model_df[target_col].notna()].copy()
        model_df["target_model"] = model_df[target_col]

    feature_list = [column for column in feature_cols if column in model_df.columns and model_df[column].notna().any()]
    if not feature_list or model_df.empty:
        return pd.DataFrame(), [], None

    for column in feature_list:
        model_df[column] = model_df[column].fillna(model_df[column].median())

    if group_col and group_col in model_df.columns:
        counts = model_df[group_col].value_counts(dropna=True)
        valid_groups = counts[counts > 1].index
        model_df = model_df[model_df[group_col].isin(valid_groups)].copy()
        if model_df.empty:
            return pd.DataFrame(), [], None
        demean_cols = ["target_model", *feature_list]
        group_means = model_df.groupby(group_col, observed=True)[demean_cols].transform("mean")
        model_df["target_model"] = model_df["target_model"] - group_means["target_model"]
        model_df[feature_list] = model_df[feature_list] - group_means[feature_list]

    std = model_df[feature_list].std(ddof=0).replace(0, np.nan)
    feature_list = [column for column in feature_list if pd.notna(std.get(column))]
    if not feature_list:
        return pd.DataFrame(), [], None

    means = model_df[feature_list].mean()
    std = model_df[feature_list].std(ddof=0).replace(0, np.nan)
    model_df[feature_list] = (model_df[feature_list] - means) / std
    model_df = model_df.dropna(subset=["target_model", *feature_list]).copy()
    if len(model_df) < 50:
        return pd.DataFrame(), [], None

    weights = None
    if weight_col and weight_col in model_df.columns:
        weights = _numeric(model_df[weight_col]).fillna(1.0).clip(lower=1.0)
    return model_df, feature_list, weights


def _fit_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    group_col: str | None = None,
    weight_col: str | None = None,
    log_target: bool = True,
) -> ModelResult:
    model_df, feature_list, weights = _prepare_regression_frame(
        df,
        target_col,
        feature_cols,
        group_col=group_col,
        weight_col=weight_col,
        log_target=log_target,
    )
    if model_df.empty or not feature_list:
        return ModelResult(pd.DataFrame(), {"n_obs": 0.0, "r_squared": np.nan})

    X = sm.add_constant(model_df[feature_list], has_constant="add")
    y = model_df["target_model"]
    model = sm.WLS(y, X, weights=weights) if weights is not None else sm.OLS(y, X)
    result = model.fit(cov_type="HC3")

    conf = result.conf_int().rename(columns={0: "ci_low", 1: "ci_high"})
    params = result.params.drop("const", errors="ignore")
    conf = conf.loc[params.index]
    if log_target:
        effect = (np.exp(params) - 1) * 100
        ci_low = (np.exp(conf["ci_low"]) - 1) * 100
        ci_high = (np.exp(conf["ci_high"]) - 1) * 100
        effect_label = "effect_pct"
    else:
        effect = params
        ci_low = conf["ci_low"]
        ci_high = conf["ci_high"]
        effect_label = "effect_value"

    coefficients = (
        pd.DataFrame(
            {
                "feature": params.index,
                effect_label: effect.to_numpy(),
                "ci_low": ci_low.to_numpy(),
                "ci_high": ci_high.to_numpy(),
                "pvalue": result.pvalues.drop("const", errors="ignore").reindex(params.index).to_numpy(),
                "tvalue": result.tvalues.drop("const", errors="ignore").reindex(params.index).to_numpy(),
            }
        )
        .assign(feature_label=lambda frame: frame["feature"].map(FEATURE_LABELS).fillna(frame["feature"]))
        .sort_values(effect_label, key=lambda series: series.abs(), ascending=False)
        .reset_index(drop=True)
    )
    metrics = {"n_obs": float(result.nobs), "r_squared": float(result.rsquared)}
    return ModelResult(coefficients, metrics)


def build_effect_chart(
    coeff_df: pd.DataFrame,
    title: str,
    *,
    effect_column: str = "effect_pct",
    xaxis_title: str = "효과(%)",
) -> go.Figure:
    if coeff_df.empty:
        return _empty_figure(f"{title} 결과가 없습니다.")

    chart_df = coeff_df.sort_values(effect_column, key=lambda series: series.abs()).copy()
    error_plus = chart_df["ci_high"] - chart_df[effect_column]
    error_minus = chart_df[effect_column] - chart_df["ci_low"]
    fig = go.Figure(
        go.Bar(
            x=chart_df[effect_column],
            y=chart_df["feature_label"],
            orientation="h",
            marker_color=np.where(chart_df[effect_column] >= 0, "#2A9D8F", "#E76F51"),
            error_x={"type": "data", "array": error_plus, "arrayminus": error_minus},
            customdata=np.stack([chart_df["pvalue"]], axis=-1),
            hovertemplate="%{y}<br>효과=%{x:.2f}<br>p-value=%{customdata[0]:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=460, xaxis_title=xaxis_title, yaxis_title="")
    return fig


def run_sale_hedonic(snapshot_df: pd.DataFrame) -> ModelResult:
    return _fit_regression(snapshot_df, "trade_price_per_m2", [*HEDONIC_FEATURES, *MISSING_FEATURES], group_col="dong_repr", weight_col="trade_count_recent")


def run_jeonse_hedonic(snapshot_df: pd.DataFrame) -> ModelResult:
    return _fit_regression(snapshot_df, "jeonse_deposit_per_m2", [*HEDONIC_FEATURES, *MISSING_FEATURES], group_col="dong_repr", weight_col="jeonse_count_recent")


def run_wolse_hedonic(snapshot_df: pd.DataFrame) -> ModelResult:
    return _fit_regression(snapshot_df, "wolse_monthly_rent_per_m2", [*HEDONIC_FEATURES, *MISSING_FEATURES], group_col="dong_repr", weight_col="wolse_count_recent")


def build_heterogeneity_frame(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty or "trade_price_per_m2" not in snapshot_df.columns:
        return pd.DataFrame()

    frame = snapshot_df.dropna(subset=["trade_price_per_m2"]).copy()
    frame["price_segment"] = _safe_qcut(frame["trade_price_per_m2"], 3, ["저가", "중가", "고가"])
    rows: list[dict[str, float | str]] = []
    feature_specs = {
        "대단지": ("household_count", True),
        "주차 우수": ("parking_per_household", True),
        "저밀도": ("floor_area_ratio", False),
        "대지지분": ("avg_land_area_per_household", True),
    }
    for segment, segment_df in frame.groupby("price_segment", observed=True):
        for label, (column, high_is_better) in feature_specs.items():
            if column not in segment_df.columns:
                continue
            valid = segment_df[[column, "trade_price_per_m2"]].dropna()
            if len(valid) < 40:
                continue
            q25, q75 = valid[column].quantile([0.25, 0.75])
            if high_is_better:
                strong = valid[valid[column] >= q75]["trade_price_per_m2"].median()
                weak = valid[valid[column] <= q25]["trade_price_per_m2"].median()
            else:
                strong = valid[valid[column] <= q25]["trade_price_per_m2"].median()
                weak = valid[valid[column] >= q75]["trade_price_per_m2"].median()
            if pd.isna(strong) or pd.isna(weak) or weak == 0:
                continue
            rows.append(
                {
                    "price_segment": segment,
                    "feature": label,
                    "premium_pct": (strong / weak - 1) * 100,
                }
            )
    return pd.DataFrame(rows)


def build_heterogeneity_chart(hetero_df: pd.DataFrame) -> go.Figure:
    if hetero_df.empty:
        return _empty_figure("분위수별 이질성 분석 결과가 없습니다.")
    fig = px.bar(
        hetero_df,
        x="price_segment",
        y="premium_pct",
        color="feature",
        barmode="group",
        title="가격 분위수별 단지 특성 프리미엄",
        labels={"price_segment": "가격 분위수", "premium_pct": "프리미엄(%)", "feature": "특성"},
    )
    fig.update_layout(height=460)
    return fig


def run_liquidity_model(yearly_df: pd.DataFrame) -> ModelResult:
    return _fit_regression(
        yearly_df,
        "annual_trade_count",
        [*HEDONIC_FEATURES, *MISSING_FEATURES],
        group_col="dong_repr",
        log_target=True,
    )


def build_liquidity_bucket_frame(yearly_df: pd.DataFrame) -> pd.DataFrame:
    if yearly_df.empty or "complex_scale_bucket" not in yearly_df.columns:
        return pd.DataFrame()

    return (
        yearly_df.groupby("complex_scale_bucket", observed=True)
        .agg(
            annual_trade_count=("annual_trade_count", "median"),
            trade_occurrence_rate=("trade_occurrence_rate", "median"),
            turnover_rate_mean=("turnover_rate_mean", "median"),
        )
        .reset_index()
    )


def build_liquidity_chart(bucket_df: pd.DataFrame) -> go.Figure:
    if bucket_df.empty:
        return _empty_figure("유동성 분석 데이터가 없습니다.")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=bucket_df["complex_scale_bucket"],
            y=bucket_df["annual_trade_count"],
            name="연간 거래건수",
            marker_color="#4C78A8",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=bucket_df["complex_scale_bucket"],
            y=bucket_df["turnover_rate_mean"],
            name="거래회전율",
            mode="lines+markers",
            line={"color": "#F58518", "width": 2.5},
        ),
        secondary_y=True,
    )
    fig.update_layout(title="세대수 구간별 유동성", height=460, hovermode="x unified")
    fig.update_yaxes(title_text="연간 거래건수", secondary_y=False)
    fig.update_yaxes(title_text="거래회전율", secondary_y=True)
    return fig


def build_rolling_coefficient_frame(yearly_df: pd.DataFrame) -> pd.DataFrame:
    if yearly_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for year, year_df in yearly_df.groupby("year", observed=True):
        result = run_sale_hedonic(year_df)
        if result.coefficients.empty:
            continue
        for feature in ROLLING_FEATURES:
            matched = result.coefficients[result.coefficients["feature"] == feature]
            if matched.empty:
                continue
            row = matched.iloc[0]
            rows.append(
                {
                    "year": int(year),
                    "feature": row["feature_label"],
                    "effect_pct": float(row["effect_pct"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )
    return pd.DataFrame(rows)


def build_rolling_coefficient_chart(rolling_df: pd.DataFrame) -> go.Figure:
    if rolling_df.empty:
        return _empty_figure("롤링 계수 데이터가 없습니다.")
    fig = px.line(
        rolling_df,
        x="year",
        y="effect_pct",
        color="feature",
        markers=True,
        title="연도별 단지 특성 프리미엄 변화",
        labels={"year": "연도", "effect_pct": "효과(%)", "feature": "특성"},
    )
    fig.update_layout(height=480)
    return fig


def run_panel_fixed_effects(panel_df: pd.DataFrame) -> ModelResult:
    if panel_df.empty:
        return ModelResult(pd.DataFrame(), {"n_obs": 0.0, "r_squared": np.nan})

    panel = panel_df.copy()
    panel["log_households"] = np.log1p(_numeric(panel.get("household_count", pd.Series(index=panel.index))).clip(lower=0))
    panel["rate_x_parking"] = _numeric(panel.get("bok_rate_change_3m", pd.Series(index=panel.index))) * _numeric(panel.get("parking_per_household", pd.Series(index=panel.index)))
    panel["rate_x_land"] = _numeric(panel.get("bok_rate_change_3m", pd.Series(index=panel.index))) * _numeric(panel.get("avg_land_area_per_household", pd.Series(index=panel.index)))
    panel["m2_x_scale"] = _numeric(panel.get("m2_yoy", pd.Series(index=panel.index))) * panel["log_households"]
    panel["m2_x_redevelop"] = _numeric(panel.get("m2_yoy", pd.Series(index=panel.index))) * _numeric(panel.get("redevelopment_option_score", pd.Series(index=panel.index)))
    features = ["bok_rate_change_3m", "m2_yoy", "rate_x_parking", "rate_x_land", "m2_x_scale", "m2_x_redevelop"]
    return _fit_regression(panel, "trade_price_per_m2_yoy", features, group_col="aptSeq", log_target=False)


def build_regime_premium_frame(panel_df: pd.DataFrame) -> pd.DataFrame:
    if panel_df.empty or "trade_price_per_m2_yoy" not in panel_df.columns:
        return pd.DataFrame()

    frame = panel_df.dropna(subset=["trade_price_per_m2_yoy", "bok_rate_change_3m", "m2_yoy"]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["rate_regime"] = np.where(frame["bok_rate_change_3m"] >= 0, "금리상승", "금리하락")
    frame["liquidity_regime"] = np.where(frame["m2_yoy"] >= frame["m2_yoy"].median(), "유동성확대", "유동성축소")
    frame["regime"] = frame["rate_regime"] + " / " + frame["liquidity_regime"]

    scale_cut = _numeric(frame.get("household_count", pd.Series(index=frame.index))).quantile(0.75)
    parking_series = _numeric(frame.get("parking_per_household", pd.Series(index=frame.index)))
    parking_cut = parking_series.dropna().quantile(0.75) if parking_series.notna().any() else np.nan
    density_cut = _numeric(frame.get("floor_area_ratio", pd.Series(index=frame.index))).quantile(0.25)

    archetypes = {
        "대단지": _numeric(frame.get("household_count", pd.Series(index=frame.index))).ge(scale_cut),
        "주차 우수": parking_series.ge(parking_cut) if pd.notna(parking_cut) else pd.Series(False, index=frame.index),
        "저밀도": _numeric(frame.get("floor_area_ratio", pd.Series(index=frame.index))).le(density_cut),
        "재건축 잠재력": _numeric(frame.get("redevelopment_option_score", pd.Series(index=frame.index))).ge(_numeric(frame.get("redevelopment_option_score", pd.Series(index=frame.index))).quantile(0.75)),
    }

    rows: list[dict[str, float | str]] = []
    for regime, regime_df in frame.groupby("regime", observed=True):
        for label, mask in archetypes.items():
            aligned_mask = mask.reindex(regime_df.index).fillna(False)
            strong = regime_df.loc[aligned_mask, "trade_price_per_m2_yoy"].mean()
            base = regime_df.loc[~aligned_mask, "trade_price_per_m2_yoy"].mean()
            if pd.isna(strong) or pd.isna(base):
                continue
            rows.append({"regime": regime, "archetype": label, "premium_pp": strong - base})
    return pd.DataFrame(rows)


def build_regime_premium_chart(regime_df: pd.DataFrame) -> go.Figure:
    if regime_df.empty:
        return _empty_figure("거시 상호작용 데이터가 없습니다.")
    fig = px.bar(
        regime_df,
        x="regime",
        y="premium_pp",
        color="archetype",
        barmode="group",
        title="거시 국면별 단지 특성 프리미엄",
        labels={"regime": "거시 국면", "premium_pp": "추가 상승률(pp)", "archetype": "유형"},
    )
    fig.update_layout(height=480)
    return fig


def build_redevelopment_frame(snapshot_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty or forecast_df.empty:
        return pd.DataFrame()

    latest_returns = (
        _ensure_datetime(forecast_df)
        .sort_values(["aptSeq", "date"])
        .drop_duplicates(subset=["aptSeq"], keep="last")[["aptSeq", "future_trade_return_12m"]]
    )
    frame = snapshot_df.merge(latest_returns, on="aptSeq", how="left")
    frame = frame.dropna(subset=["redevelopment_option_score"]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["score_bucket"] = _safe_qcut(frame["redevelopment_option_score"], 4, ["낮음", "중간", "높음", "매우 높음"])
    grouped = (
        frame.groupby("score_bucket", observed=True)
        .agg(
            complex_count=("aptSeq", "nunique"),
            trade_price_per_m2=("trade_price_per_m2", "median"),
            jeonse_ratio=("jeonse_ratio", "median"),
            future_trade_return_12m=("future_trade_return_12m", "median"),
        )
        .reset_index()
    )
    return grouped


def build_redevelopment_chart(redevelopment_df: pd.DataFrame) -> go.Figure:
    if redevelopment_df.empty:
        return _empty_figure("재건축 옵션 분석 데이터가 없습니다.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=redevelopment_df["score_bucket"],
            y=redevelopment_df["trade_price_per_m2"],
            name="현재 매매가",
            marker_color="#6C5B7B",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=redevelopment_df["score_bucket"],
            y=redevelopment_df["future_trade_return_12m"],
            name="향후 12개월 수익률",
            mode="lines+markers",
            line={"color": "#F67280", "width": 2.5},
        ),
        secondary_y=True,
    )
    fig.update_layout(title="재건축 옵션 점수와 가격/수익률", height=480, hovermode="x unified")
    fig.update_yaxes(title_text="중위 m2당 매매가", secondary_y=False)
    fig.update_yaxes(title_text="향후 12개월 수익률(%)", secondary_y=True)
    return fig


def build_spillover_frame(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if panel_df.empty:
        return pd.DataFrame(), {"lag1_corr": np.nan, "spread_pp": np.nan}

    frame = panel_df.dropna(subset=["dong_repr", "date", "trade_price_per_m2_yoy"]).copy()
    if frame.empty:
        return pd.DataFrame(), {"lag1_corr": np.nan, "spread_pp": np.nan}

    scale_cut = _numeric(frame.get("household_count", pd.Series(index=frame.index))).quantile(0.75)
    parking_series = _numeric(frame.get("parking_per_household", pd.Series(index=frame.index)))
    parking_cut = parking_series.dropna().quantile(0.75) if parking_series.notna().any() else np.nan
    leader_mask = _numeric(frame.get("household_count", pd.Series(index=frame.index))).ge(scale_cut)
    if pd.notna(parking_cut):
        leader_mask = leader_mask | parking_series.ge(parking_cut)
    frame["leader_flag"] = leader_mask

    grouped = (
        frame.groupby(["dong_repr", "date", "leader_flag"], observed=True)["trade_price_per_m2_yoy"]
        .mean()
        .unstack("leader_flag")
        .rename(columns={False: "follower_yoy", True: "leader_yoy"})
        .dropna(subset=["leader_yoy", "follower_yoy"])
        .reset_index()
        .sort_values(["dong_repr", "date"])
    )
    if grouped.empty:
        return pd.DataFrame(), {"lag1_corr": np.nan, "spread_pp": np.nan}

    grouped["leader_yoy_lag1"] = grouped.groupby("dong_repr", observed=True)["leader_yoy"].shift(1)
    corr_df = grouped.dropna(subset=["leader_yoy_lag1", "follower_yoy"])
    lag1_corr = float(corr_df["leader_yoy_lag1"].corr(corr_df["follower_yoy"])) if not corr_df.empty else np.nan
    spread = float((grouped["leader_yoy"] - grouped["follower_yoy"]).mean())
    monthly = grouped.groupby("date", observed=True)[["leader_yoy", "follower_yoy"]].mean().reset_index()
    return monthly, {"lag1_corr": lag1_corr, "spread_pp": spread}


def build_spillover_chart(spillover_df: pd.DataFrame) -> go.Figure:
    if spillover_df.empty:
        return _empty_figure("확산 분석 데이터가 없습니다.")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spillover_df["date"],
            y=spillover_df["leader_yoy"],
            mode="lines",
            name="선도 단지 YoY",
            line={"color": "#264653", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spillover_df["date"],
            y=spillover_df["follower_yoy"],
            mode="lines",
            name="주변 단지 YoY",
            line={"color": "#E76F51", "width": 2.5},
        )
    )
    fig.update_layout(title="대단지/주차우수 단지의 확산 효과", height=460, yaxis_title="YoY(%)", hovermode="x unified")
    return fig


def _forecast_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "log_households",
        "parking_per_household",
        "floor_area_ratio",
        "building_coverage_ratio",
        "avg_land_area_per_household",
        "complex_age",
        "redevelopment_option_score",
        "trade_price_per_m2_lag1",
        "trade_price_per_m2_lag3",
        "trade_price_per_m2_lag6",
        "trade_price_per_m2_lag12",
        "jeonse_deposit_per_m2_lag1",
        "jeonse_deposit_per_m2_lag3",
        "wolse_monthly_rent_per_m2_lag1",
        "wolse_monthly_rent_per_m2_lag3",
        "trade_count_lag1",
        "trade_count_lag3",
        "jeonse_ratio_lag1",
        "jeonse_ratio_lag3",
        "conversion_rate_lag1",
        "conversion_rate_lag3",
        "bok_rate",
        "bok_rate_change_3m",
        "m2_yoy",
        "usdkrw",
        *MISSING_FEATURES,
    ]
    return [column for column in preferred if column in df.columns]


def _prepare_forecast_frame(forecast_df: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_datetime(forecast_df)
    frame["log_households"] = np.log1p(_numeric(frame.get("household_count", pd.Series(index=frame.index))).clip(lower=0))
    if "date" in frame.columns:
        frame["year"] = frame["date"].dt.year
    if "completion_year" in frame.columns and "year" in frame.columns:
        frame["complex_age"] = frame["year"] - _numeric(frame["completion_year"])
        frame.loc[frame["complex_age"] < 0, "complex_age"] = np.nan
    return frame


def _run_forecast_model(
    forecast_df: pd.DataFrame,
    target_col: str,
    *,
    base_col: str | None = None,
    log_target: bool = True,
    test_periods: int = 6,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    frame = _prepare_forecast_frame(forecast_df)
    feature_cols = _forecast_feature_columns(frame)
    required = _existing_columns(frame, [target_col, "ym", "date", *feature_cols])
    model_df = frame[required].copy()
    if target_col not in model_df.columns or "ym" not in model_df.columns or "date" not in model_df.columns:
        return pd.DataFrame(), {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_test": 0.0}, pd.DataFrame()

    model_df[target_col] = _numeric(model_df[target_col])
    for column in feature_cols:
        model_df[column] = _numeric(model_df[column]).replace([np.inf, -np.inf], np.nan)
    if log_target:
        model_df = model_df[model_df[target_col] > 0].copy()
    else:
        model_df = model_df[model_df[target_col].notna()].copy()
    if model_df.empty:
        return pd.DataFrame(), {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_test": 0.0}, pd.DataFrame()

    ym_values = sorted(model_df["ym"].dropna().astype(str).unique())
    if len(ym_values) < 8:
        return pd.DataFrame(), {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_test": 0.0}, pd.DataFrame()

    test_periods = min(test_periods, max(1, len(ym_values) // 4))
    train_ym = ym_values[:-test_periods]
    test_ym = ym_values[-test_periods:]
    train_df = model_df[model_df["ym"].isin(train_ym)].copy()
    test_df = model_df[model_df["ym"].isin(test_ym)].copy()
    if train_df.empty or test_df.empty:
        return pd.DataFrame(), {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan, "n_test": 0.0}, pd.DataFrame()

    usable_features = [column for column in feature_cols if column in train_df.columns and train_df[column].notna().any()]
    if not usable_features:
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
    y_train = np.log(train_df[target_col]).astype(float) if log_target else train_df[target_col].astype(float)
    model = sm.OLS(y_train, sm.add_constant(train_X, has_constant="add")).fit(cov_type="HC3")
    pred_values = model.predict(sm.add_constant(test_X, has_constant="add"))
    predictions = np.exp(pred_values) if log_target else pred_values

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
    mape_base = pred_df["actual"].replace(0, np.nan)
    mape = float((np.abs(residual) / mape_base).dropna().mean() * 100) if mape_base.notna().any() else np.nan

    directional_accuracy = np.nan
    if base_col and base_col in test_df.columns:
        direction_df = test_df[[target_col, base_col]].copy()
        direction_df["prediction"] = predictions
        actual_direction = np.sign(direction_df[target_col] - direction_df[base_col])
        pred_direction = np.sign(direction_df["prediction"] - direction_df[base_col])
        valid = actual_direction.notna() & pred_direction.notna()
        if valid.any():
            directional_accuracy = float((actual_direction[valid] == pred_direction[valid]).mean() * 100)
    elif not log_target:
        valid = pred_df["actual"].notna() & pred_df["prediction"].notna()
        if valid.any():
            directional_accuracy = float((np.sign(pred_df.loc[valid, "actual"]) == np.sign(pred_df.loc[valid, "prediction"])).mean() * 100)

    params = model.params.drop("const", errors="ignore")
    importance = (
        pd.DataFrame({"feature": params.index, "coefficient": params.to_numpy()})
        .assign(feature_label=lambda frame: frame["feature"].map(FEATURE_LABELS).fillna(frame["feature"]))
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "n_test": float(len(test_df)),
    }
    return pred_df, metrics, importance


def run_sale_forecast(forecast_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    return _run_forecast_model(forecast_df, f"trade_price_per_m2_t{horizon}", base_col="trade_price_per_m2_lag1")


def run_rent_forecast(
    forecast_df: pd.DataFrame,
    metric: str,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    target = "jeonse_deposit_per_m2" if metric == "jeonse" else "wolse_monthly_rent_per_m2"
    return _run_forecast_model(forecast_df, f"{target}_t{horizon}", base_col=f"{target}_lag1")


def run_return_forecast(
    forecast_df: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    target_map = {
        "trade": "future_trade_return_12m",
        "jeonse": "future_jeonse_return_12m",
        "wolse": "future_wolse_return_12m",
    }
    return _run_forecast_model(forecast_df, target_map[metric], log_target=False, test_periods=12)


def run_ratio_forecast(
    forecast_df: pd.DataFrame,
    metric: str,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    target = "jeonse_ratio" if metric == "jeonse_ratio" else "conversion_rate"
    return _run_forecast_model(forecast_df, f"{target}_t{horizon}", base_col=f"{target}_lag1")


def build_forecast_chart(pred_df: pd.DataFrame, title: str, yaxis_title: str) -> go.Figure:
    if pred_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["actual"],
            mode="lines+markers",
            name="실제",
            line={"color": "#264653", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["prediction"],
            mode="lines+markers",
            name="예측",
            line={"color": "#E76F51", "width": 2.5, "dash": "dash"},
        )
    )
    fig.update_layout(title=title, height=460, yaxis_title=yaxis_title, hovermode="x unified")
    return fig


def build_importance_chart(importance_df: pd.DataFrame, title: str) -> go.Figure:
    if importance_df.empty:
        return _empty_figure(f"{title} 데이터가 없습니다.")
    top = importance_df.head(10).sort_values("abs_coefficient")
    fig = go.Figure(
        go.Bar(
            x=top["abs_coefficient"],
            y=top["feature_label"],
            orientation="h",
            marker_color="#6C5B7B",
        )
    )
    fig.update_layout(title=title, height=420, xaxis_title="절대계수", yaxis_title="")
    return fig


def build_scenario_frame(
    forecast_df: pd.DataFrame,
    *,
    rate_delta: float,
    liquidity_delta: float,
    supply_delta: float,
) -> pd.DataFrame:
    prepared = _prepare_forecast_frame(forecast_df)
    training = prepared.dropna(subset=["future_trade_return_12m"]).copy()
    if training.empty:
        return pd.DataFrame()

    feature_cols = [
        "log_households",
        "parking_per_household",
        "floor_area_ratio",
        "avg_land_area_per_household",
        "redevelopment_option_score",
        "bok_rate_change_3m",
        "m2_yoy",
    ]
    existing = [column for column in feature_cols if column in training.columns and training[column].notna().any()]
    if len(existing) < 4:
        return pd.DataFrame()
    train = training[["future_trade_return_12m", *existing]].copy()
    for column in existing:
        numeric_values = _numeric(train[column])
        train[column] = numeric_values.fillna(numeric_values.median())
    train = train.replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < 100:
        return pd.DataFrame()

    X = sm.add_constant(train[existing], has_constant="add")
    y = train["future_trade_return_12m"]
    model = sm.OLS(y, X).fit(cov_type="HC3")

    medians = train[existing].median()
    quantiles = train[existing].quantile([0.1, 0.5, 0.9])
    archetypes = {
        "기준": medians.to_dict(),
        "대단지": {**medians.to_dict(), "log_households": float(quantiles.loc[0.9, "log_households"])},
        "주차 우수": {**medians.to_dict(), "parking_per_household": float(quantiles.loc[0.9, "parking_per_household"]) if "parking_per_household" in quantiles.columns else float(medians.get("parking_per_household", 0.0))},
        "저밀도": {**medians.to_dict(), "floor_area_ratio": float(quantiles.loc[0.1, "floor_area_ratio"])},
        "재건축 잠재력": {**medians.to_dict(), "redevelopment_option_score": float(quantiles.loc[0.9, "redevelopment_option_score"])},
    }

    mean_scale = float(train["log_households"].std(ddof=0) or 1.0) if "log_households" in train.columns else 1.0
    mean_density = float(train["floor_area_ratio"].std(ddof=0) or 1.0) if "floor_area_ratio" in train.columns else 1.0
    rows: list[dict[str, float | str]] = []
    for label in ARCHETYPE_ORDER:
        feature_row = archetypes[label]
        base_row = feature_row.copy()
        scenario_row = feature_row.copy()
        if "bok_rate_change_3m" in scenario_row:
            scenario_row["bok_rate_change_3m"] = float(scenario_row["bok_rate_change_3m"]) + rate_delta
        if "m2_yoy" in scenario_row:
            scenario_row["m2_yoy"] = float(scenario_row["m2_yoy"]) + liquidity_delta

        base_pred = float(model.predict(sm.add_constant(pd.DataFrame([base_row])[existing], has_constant="add")).iloc[0])
        scenario_pred = float(model.predict(sm.add_constant(pd.DataFrame([scenario_row])[existing], has_constant="add")).iloc[0])
        density_exposure = (float(feature_row.get("floor_area_ratio", medians.get("floor_area_ratio", 0.0))) - float(medians.get("floor_area_ratio", 0.0))) / mean_density
        scale_exposure = (float(feature_row.get("log_households", medians.get("log_households", 0.0))) - float(medians.get("log_households", 0.0))) / mean_scale
        supply_overlay = -0.6 * supply_delta * (0.6 * density_exposure + 0.4 * scale_exposure)
        rows.append(
            {
                "archetype": label,
                "base_return": base_pred,
                "scenario_return": scenario_pred + supply_overlay,
                "incremental_return": scenario_pred + supply_overlay - base_pred,
            }
        )
    return pd.DataFrame(rows)


def build_scenario_chart(scenario_df: pd.DataFrame) -> go.Figure:
    if scenario_df.empty:
        return _empty_figure("시나리오 분석 데이터가 없습니다.")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=scenario_df["archetype"],
            y=scenario_df["incremental_return"],
            marker_color=np.where(scenario_df["incremental_return"] >= 0, "#2A9D8F", "#E76F51"),
            name="추가 수익률",
        )
    )
    fig.update_layout(title="시나리오별 단지 유형 반응", height=440, yaxis_title="추가 수익률(%)")
    return fig
