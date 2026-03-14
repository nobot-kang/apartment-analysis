"""Level 4 고급 분석."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import fcluster, linkage

from analysis.common import (
    ANALYSIS_START_YM,
    SEOUL_DISTRICT_COORDS,
    add_seoul_coordinates,
    aggregate_trade_scope,
    load_macro_monthly_df,
    load_trade_detail_df,
    load_trade_summary_df,
    optional_import,
)

DID_EVENTS = {
    "GTX-A 착공 발표": {
        "event_ym": "202012",
        "treatment_regions": ["고양시 덕양구", "파주시"],
        "control_regions": ["의정부시", "구리시"],
        "description": "GTX-A 착공 발표(2020-12) 전후 가격 효과",
    },
    "GTX-B 예타 통과": {
        "event_ym": "202106",
        "treatment_regions": ["남양주시", "하남시"],
        "control_regions": ["부천시 원미구", "의왕시"],
        "description": "GTX-B 예타 통과(2021-06) 전후 가격 효과",
    },
}
PHASE_COLORS = {
    "과열": "#D1495B",
    "회복": "#F4A261",
    "조정": "#5C677D",
    "침체": "#2A9D8F",
}


def _add_datetime_event_marker(fig: go.Figure, event_date: pd.Timestamp, label: str) -> None:
    """Plotly의 datetime 축 add_vline 주석 버그를 우회해 이벤트 마커를 추가한다."""
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


def _build_scope_frame(region_codes: list[str] | None = None, scope_name: str | None = None) -> pd.DataFrame:
    trade_scope = aggregate_trade_scope(load_trade_summary_df(ANALYSIS_START_YM), region_codes, scope_name)
    macro = load_macro_monthly_df(ANALYSIS_START_YM)
    if trade_scope.empty or macro.empty:
        return pd.DataFrame()
    return trade_scope.merge(macro, on=["ym", "date"], how="inner").sort_values("date").reset_index(drop=True)


def build_prediction_dataset(region_codes: list[str] | None = None, scope_name: str | None = None) -> pd.DataFrame:
    """다음 달 평균 매매가 예측용 데이터셋을 생성한다."""
    df = _build_scope_frame(region_codes, scope_name)
    if df.empty:
        return df

    result = df.copy()
    result["target"] = result["평균거래금액"].shift(-1)
    for lag in [1, 3, 6, 12]:
        result[f"price_lag_{lag}"] = result["평균거래금액"].shift(lag)
        result[f"volume_lag_{lag}"] = result["거래건수"].shift(lag)
    if "bok_rate" in result.columns:
        result["rate_change_3m"] = result["bok_rate"].diff(3)
    if "m2" in result.columns:
        result["m2_yoy"] = result["m2"].pct_change(12) * 100
    if "usdkrw" in result.columns:
        result["fx_yoy"] = result["usdkrw"].pct_change(12) * 100
    return result.dropna().reset_index(drop=True)


def run_prediction_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """단순 시계열 회귀 예측과 성능 지표를 계산한다."""
    empty_metrics = {"rmse": math.nan, "mape": math.nan, "r_squared": math.nan}
    if df.empty:
        return pd.DataFrame(), empty_metrics, []

    feature_cols = [
        column
        for column in df.columns
        if column.startswith(("price_lag_", "volume_lag_")) or column in {"bok_rate", "rate_change_3m", "m2_yoy", "fx_yoy"}
    ]
    feature_cols = [column for column in feature_cols if column in df.columns]
    if not feature_cols:
        return pd.DataFrame(), empty_metrics, []

    min_train_size = max(len(feature_cols) + 2, 12)
    if len(df) <= min_train_size:
        return pd.DataFrame(), empty_metrics, feature_cols

    split_index = max(int(len(df) * 0.8), min_train_size)
    split_index = min(split_index, len(df) - 1)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    if len(train) <= len(feature_cols) + 1 or test.empty:
        return pd.DataFrame(), empty_metrics, feature_cols

    x_train = sm.add_constant(train[feature_cols], has_constant="add")
    x_test = sm.add_constant(test[feature_cols], has_constant="add")
    model = sm.OLS(np.log(train["target"]), x_train).fit()
    test["predicted"] = np.exp(model.predict(x_test))
    test["actual"] = test["target"]
    test["scope_name"] = df.get("scope_name", pd.Series(["선택 지역"] * len(df))).iloc[0]

    rmse = float(np.sqrt(np.mean((test["actual"] - test["predicted"]) ** 2)))
    mape = float(np.mean(np.abs((test["actual"] - test["predicted"]) / test["actual"])) * 100)
    return test[["date", "actual", "predicted", "scope_name"]], {"rmse": rmse, "mape": mape, "r_squared": float(model.rsquared)}, feature_cols


def build_prediction_chart(pred_df: pd.DataFrame, scope_name: str) -> go.Figure:
    """실제값과 예측값을 비교한다."""
    fig = go.Figure()
    if pred_df.empty:
        fig.update_layout(title="예측 결과 데이터 없음", height=420)
        return fig

    fig.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["actual"], name="실제값", line={"color": "#D1495B", "width": 2.5}))
    fig.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["predicted"], name="예측값", line={"color": "#457B9D", "dash": "dash"}))
    fig.update_layout(title=f"{scope_name} 다음 달 매매가 예측", yaxis_title="평균 매매가 (만원)", height=430, hovermode="x unified")
    return fig


def _dtw_distance(series_a: np.ndarray, series_b: np.ndarray) -> float:
    """의존성 없이 DTW 거리를 계산한다."""
    n_rows, n_cols = len(series_a), len(series_b)
    cost = np.full((n_rows + 1, n_cols + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            dist = abs(series_a[i - 1] - series_b[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n_rows, n_cols])


def load_cluster_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """서울 자치구 가격 시계열 행렬을 준비한다."""
    trade = load_trade_summary_df(ANALYSIS_START_YM)
    if trade.empty:
        return pd.DataFrame(), pd.DataFrame()

    seoul = trade[trade["_lawd_cd"].astype(str).str.startswith("11")].copy()
    pivot = seoul.pivot_table(index="date", columns="_region_name", values="평균거래금액", aggfunc="mean").sort_index()
    pivot = pivot.interpolate(limit_direction="both").ffill().bfill()
    scaled = (pivot - pivot.mean()) / pivot.std(ddof=0).replace(0, np.nan)
    scaled = scaled.fillna(0)
    return pivot, scaled


def run_dtw_clustering(n_clusters: int = 4) -> pd.DataFrame:
    """서울 자치구를 DTW 거리로 군집화한다."""
    _, scaled = load_cluster_dataset()
    if scaled.empty:
        return pd.DataFrame(columns=["district", "cluster", "cluster_name", "lat", "lon"])

    districts = list(scaled.columns)
    if len(districts) == 1:
        cluster_df = pd.DataFrame({"district": districts, "cluster": [1]})
    else:
        condensed = []
        for i in range(len(districts)):
            for j in range(i + 1, len(districts)):
                condensed.append(_dtw_distance(scaled.iloc[:, i].to_numpy(), scaled.iloc[:, j].to_numpy()))
        linkage_matrix = linkage(np.asarray(condensed), method="average")
        labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
        cluster_df = pd.DataFrame({"district": districts, "cluster": labels})

    cluster_df["cluster_name"] = cluster_df["cluster"].map(lambda value: f"Cluster {value}")
    reverse_map = {
        name: code
        for code, name in load_trade_summary_df(ANALYSIS_START_YM)[["_lawd_cd", "_region_name"]].drop_duplicates().values
    }
    cluster_df["_lawd_cd"] = cluster_df["district"].map(reverse_map)
    cluster_df = add_seoul_coordinates(cluster_df)
    return cluster_df.reset_index(drop=True)


def build_cluster_heatmap() -> go.Figure:
    """군집화 전 표준화된 가격 패턴 히트맵을 그린다."""
    _, scaled = load_cluster_dataset()
    cluster_df = run_dtw_clustering()
    if scaled.empty or cluster_df.empty:
        return go.Figure()

    ordered = cluster_df.sort_values(["cluster", "district"])["district"].tolist()
    matrix = scaled[ordered].T
    fig = go.Figure(data=go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, colorscale="RdBu_r", zmid=0))
    fig.update_layout(title="서울 자치구 DTW 군집 히트맵", height=720, xaxis_title="월", yaxis_title="지역")
    return fig


def build_cluster_map(cluster_df: pd.DataFrame) -> go.Figure:
    """군집 결과를 서울 지도 위에 표시한다."""
    if cluster_df.empty:
        return go.Figure()

    fig = px.scatter_mapbox(
        cluster_df,
        lat="lat",
        lon="lon",
        color="cluster_name",
        hover_name="district",
        zoom=10,
        center={"lat": 37.5665, "lon": 126.9780},
        mapbox_style="open-street-map",
        title="서울 자치구 DTW 군집 지도",
    )
    fig.update_layout(height=540, margin={"l": 0, "r": 0, "t": 50, "b": 0})
    return fig


def load_anomaly_data(region_code: str, years: list[int] | None = None) -> pd.DataFrame:
    """이상거래 탐지용 상세 데이터를 불러온다."""
    df = load_trade_detail_df(
        years=years,
        region_codes=[region_code],
        columns=["date", "price", "area", "floor", "age", "apt_name_repr", "dong_repr"],
    )
    if df.empty:
        return df

    result = df.copy()
    result["price_per_m2"] = result["price"] / result["area"].replace(0, pd.NA)
    return result.dropna(subset=["price_per_m2", "area", "floor"])


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """설치 환경에 따라 IsolationForest 또는 강건 z-score를 사용한다."""
    if df.empty:
        return df.copy()

    result = df.copy()
    sklearn_ensemble = optional_import("sklearn.ensemble")
    feature_frame = result[["price_per_m2", "area", "floor", "age"]]
    features = feature_frame.fillna(feature_frame.median())

    if sklearn_ensemble is not None:
        model = sklearn_ensemble.IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
        model.fit(features)
        result["anomaly_score"] = model.score_samples(features)
        result["is_anomaly"] = model.predict(features) == -1
        return result

    medians = features.median()
    mads = (features - medians).abs().median().replace(0, np.nan)
    robust_z = ((features - medians).abs() / mads).fillna(0)
    result["anomaly_score"] = robust_z.mean(axis=1)
    threshold = result["anomaly_score"].quantile(0.97)
    result["is_anomaly"] = result["anomaly_score"] >= threshold
    return result


def build_anomaly_chart(df: pd.DataFrame, scope_name: str) -> go.Figure:
    """이상거래를 시계열 산점도로 표시한다."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="이상거래 데이터 없음", height=480)
        return fig

    normal = df[~df["is_anomaly"]]
    anomaly = df[df["is_anomaly"]]
    fig.add_trace(go.Scatter(x=normal["date"], y=normal["price_per_m2"], mode="markers", name="정상", marker={"size": 5, "color": "#A8DADC", "opacity": 0.5}))
    fig.add_trace(
        go.Scatter(
            x=anomaly["date"],
            y=anomaly["price_per_m2"],
            mode="markers",
            name="이상",
            marker={"size": 9, "color": "#D1495B", "symbol": "x"},
            text=anomaly["apt_name_repr"],
            hovertemplate="%{text}<br>㎡당 %{y:,.0f}만원<extra></extra>",
        )
    )
    fig.update_layout(title=f"{scope_name} 이상거래 탐지", xaxis_title="거래일", yaxis_title="㎡당 가격 (만원)", height=500)
    return fig


def build_did_dataset(event_key: str) -> pd.DataFrame:
    """GTX 이벤트별 Difference-in-Differences 데이터셋을 만든다."""
    event = DID_EVENTS[event_key]
    trade = load_trade_summary_df(ANALYSIS_START_YM)
    macro = load_macro_monthly_df(ANALYSIS_START_YM)
    if trade.empty or macro.empty:
        return pd.DataFrame()

    all_regions = event["treatment_regions"] + event["control_regions"]
    subset = trade[trade["_region_name"].isin(all_regions)].copy()
    if subset.empty:
        return pd.DataFrame()

    subset = subset.merge(macro[["ym", "bok_rate"]], on="ym", how="left")
    subset["treatment"] = subset["_region_name"].isin(event["treatment_regions"]).astype(int)
    subset["post"] = (subset["ym"] >= event["event_ym"]).astype(int)
    subset["log_price"] = np.log(subset["평균거래금액"])

    months = sorted(subset["ym"].unique())
    event_index = months.index(event["event_ym"]) if event["event_ym"] in months else len(months) // 2
    window = months[max(0, event_index - 12) : min(len(months), event_index + 13)]
    return subset[subset["ym"].isin(window)].copy()


def run_did_regression(df: pd.DataFrame) -> dict[str, float | str | list[float]]:
    """DiD 회귀를 수행하고 핵심 통계를 반환한다."""
    if df.empty:
        return {"summary": "데이터 없음", "pct_effect": math.nan, "did_pvalue": math.nan, "did_ci": [math.nan, math.nan], "n_obs": 0}

    result = smf.ols("log_price ~ treatment + post + treatment:post + bok_rate", data=df).fit(cov_type="HC3")
    coef = result.params.get("treatment:post", math.nan)
    ci = result.conf_int().loc["treatment:post"].tolist() if "treatment:post" in result.params.index else [math.nan, math.nan]
    pct_effect = (np.exp(coef) - 1) * 100 if pd.notna(coef) else math.nan
    return {
        "summary": result.summary().as_text(),
        "pct_effect": float(pct_effect) if pd.notna(pct_effect) else math.nan,
        "did_pvalue": float(result.pvalues.get("treatment:post", math.nan)),
        "did_ci": [float(ci[0]), float(ci[1])],
        "n_obs": int(result.nobs),
    }


def build_parallel_trend_chart(df: pd.DataFrame, event_ym: str) -> go.Figure:
    """처리군과 대조군의 이벤트 전후 평균 가격 추이를 비교한다."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="DiD 비교 데이터 없음", height=420)
        return fig

    working = df[["date", "treatment", "평균거래금액", "거래건수"]].copy()
    working["weighted_price"] = working["평균거래금액"] * working["거래건수"]
    trend = (
        working.groupby(["date", "treatment"], observed=True)
        .agg(weighted_price=("weighted_price", "sum"), trade_count=("거래건수", "sum"))
        .reset_index()
    )
    trend = trend[trend["trade_count"] > 0].copy()
    trend["avg_price"] = trend["weighted_price"] / trend["trade_count"]

    mapping = {1: ("처리군", "#D1495B"), 0: ("대조군", "#457B9D")}
    for treatment, (label, color) in mapping.items():
        subset = trend[trend["treatment"] == treatment]
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["avg_price"],
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 2.5},
            )
        )

    _add_datetime_event_marker(fig, pd.Timestamp(f"{event_ym[:4]}-{event_ym[4:]}-01"), "이벤트")
    fig.update_layout(title="처리군 vs 대조군 평행 추세", yaxis_title="평균 매매가 (만원)", height=430, hovermode="x unified")
    return fig


def load_cycle_features() -> pd.DataFrame:
    """시장 사이클 분류용 특징량을 생성한다."""
    seoul_codes = list(SEOUL_DISTRICT_COORDS.keys())
    df = _build_scope_frame(seoul_codes, "서울 전체")
    if df.empty:
        return df

    result = df.copy()
    result["rate_direction"] = np.sign(result["bok_rate"].diff()) if "bok_rate" in result.columns else 0
    result["m2_yoy"] = result["m2"].pct_change(12) * 100 if "m2" in result.columns else np.nan
    result["vol_mom"] = result["거래건수"].pct_change() * 100
    result["vol_mom_3ma"] = result["vol_mom"].rolling(3).mean()
    result["price_yoy"] = result["평균거래금액"].pct_change(12) * 100

    def classify(row: pd.Series) -> str:
        if row.get("rate_direction", 0) <= 0 and row.get("m2_yoy", 0) >= 5 and row.get("vol_mom_3ma", 0) >= 0:
            return "과열"
        if row.get("rate_direction", 0) >= 1 and row.get("m2_yoy", 0) <= 2 and row.get("vol_mom_3ma", 0) <= -5:
            return "침체"
        if row.get("rate_direction", 0) >= 1 and row.get("vol_mom_3ma", 0) <= 0:
            return "조정"
        return "회복"

    result["phase_rule"] = result.apply(classify, axis=1)
    return result.dropna(subset=["price_yoy", "m2_yoy", "vol_mom_3ma"]).reset_index(drop=True)


def build_cycle_dashboard(df: pd.DataFrame, use_hmm: bool = False) -> go.Figure:
    """서울 시장 사이클 대시보드 차트를 생성한다."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        vertical_spacing=0.04,
        subplot_titles=("서울 평균 매매가와 국면", "금리 방향", "M2 YoY", "거래량 MoM(3MA)"),
    )
    if df.empty:
        fig.update_layout(title="시장 사이클 데이터 없음", height=780)
        return fig

    hmm_module = optional_import("hmmlearn.hmm")
    working = df.copy()
    working["phase"] = working["phase_rule"]
    if use_hmm and hmm_module is not None:
        model = hmm_module.GaussianHMM(n_components=4, covariance_type="full", n_iter=200, random_state=42)
        features = working[["rate_direction", "m2_yoy", "vol_mom_3ma"]].to_numpy()
        model.fit(features)
        states = model.predict(features)
        working["phase"] = pd.Series(states).map({0: "조정", 1: "회복", 2: "침체", 3: "과열"}).fillna(working["phase_rule"])

    fig.add_trace(go.Scatter(x=working["date"], y=working["평균거래금액"], line={"color": "#1D3557", "width": 2.5}, name="평균 매매가"), row=1, col=1)
    phase_starts = working.index[working["phase"] != working["phase"].shift()].tolist()
    phase_ends = phase_starts[1:] + [len(working) - 1]
    for start, end in zip(phase_starts, phase_ends):
        phase = working.loc[start, "phase"]
        fig.add_vrect(
            x0=working.loc[start, "date"],
            x1=working.loc[end, "date"],
            fillcolor=PHASE_COLORS.get(phase, "#CCCCCC"),
            opacity=0.15,
            line_width=0,
            row=1,
            col=1,
        )
    fig.add_trace(go.Bar(x=working["date"], y=working["rate_direction"], marker_color="#457B9D", name="금리 방향"), row=2, col=1)
    fig.add_trace(go.Scatter(x=working["date"], y=working["m2_yoy"], line={"color": "#F4A261"}, name="M2 YoY"), row=3, col=1)
    fig.add_trace(go.Bar(x=working["date"], y=working["vol_mom_3ma"], marker_color="#2A9D8F", name="거래량 MoM"), row=4, col=1)
    fig.update_layout(title=f"서울 부동산 시장 국면 분류 ({'HMM' if use_hmm and hmm_module is not None else '규칙 기반'})", height=820, showlegend=False)
    return fig


def get_current_phase(df: pd.DataFrame) -> dict[str, float | str]:
    """최근 시점의 시장 국면 정보를 반환한다."""
    if df.empty:
        return {"phase": "데이터 없음", "color": "#999999", "year_month": "N/A", "bok_rate": math.nan, "m2_yoy": math.nan, "vol_mom": math.nan}

    latest = df.sort_values("date").iloc[-1]
    return {
        "phase": latest["phase_rule"],
        "color": PHASE_COLORS.get(latest["phase_rule"], "#999999"),
        "year_month": latest["ym"],
        "bok_rate": float(latest.get("bok_rate", math.nan)),
        "m2_yoy": float(latest.get("m2_yoy", math.nan)),
        "vol_mom": float(latest.get("vol_mom", math.nan)),
    }
