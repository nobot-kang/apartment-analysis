"""Page 05 - Level 4 고급 분석."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import get_scope_codes
from analysis.level4 import (
    DID_EVENTS,
    build_anomaly_chart,
    build_cluster_heatmap,
    build_cluster_map,
    build_cycle_dashboard,
    build_did_dataset,
    build_parallel_trend_chart,
    build_prediction_chart,
    build_prediction_dataset,
    build_scope_frame,
    get_current_phase,
    load_cluster_dataset,
    run_did_regression,
    run_dtw_clustering,
    run_prediction_model,
)
from dashboard.data_loader import (
    get_filtered_trade_anomalies,
    get_scope_option_list,
    load_cycle_features_precomputed,
    load_macro_monthly,
    load_trade_summary,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_prediction(scope_name: str):
    trade_df = load_trade_summary()
    macro_df = load_macro_monthly()
    scope_frame = build_scope_frame(trade_df, macro_df, get_scope_codes(scope_name), scope_name)
    dataset = build_prediction_dataset(scope_frame)
    return run_prediction_model(dataset)


@st.cache_resource(show_spinner=False)
def _get_cluster_base():
    trade_df = load_trade_summary()
    return load_cluster_dataset(trade_df)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_clusters(n_clusters: int):
    _pivot, scaled, region_lookup = _get_cluster_base()
    return run_dtw_clustering(scaled, region_lookup, n_clusters)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_did(event_key: str):
    trade_df = load_trade_summary()
    macro_df = load_macro_monthly()
    did_df = build_did_dataset(trade_df, macro_df, event_key)
    return did_df, run_did_regression(did_df)


def _render_phase_banner() -> None:
    cycle_df = load_cycle_features_precomputed()
    phase = get_current_phase(cycle_df)
    st.markdown(
        f"""
        <div style="padding:12px 16px;border-left:6px solid {phase['color']};background:{phase['color']}18;border-radius:8px;margin-bottom:16px;">
            <strong>현재 시장 국면:</strong> {phase['phase']}<br/>
            <span>기준월 {phase['year_month']} | 기준금리 {phase['bok_rate']:.2f}% | M2 YoY {phase['m2_yoy']:.2f}% | 거래량 MoM {phase['vol_mom']:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction() -> None:
    st.header("Level 4 - 가격 예측")
    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    _render_phase_banner()
    scope_name = st.selectbox("예측 범위", get_scope_option_list(), index=0, key="level4_prediction_scope")
    pred_df, metrics, feature_cols = _get_prediction(scope_name)
    st.plotly_chart(build_prediction_chart(pred_df, scope_name), width="stretch")
    cols = st.columns(3)
    cols[0].metric("RMSE", f"{metrics['rmse']:,.0f}" if metrics["rmse"] == metrics["rmse"] else "N/A")
    cols[1].metric("MAPE", f"{metrics['mape']:.2f}%" if metrics["mape"] == metrics["mape"] else "N/A")
    cols[2].metric("R²", f"{metrics['r_squared']:.4f}" if metrics["r_squared"] == metrics["r_squared"] else "N/A")
    if feature_cols:
        st.caption("사용 특징량: " + ", ".join(feature_cols))


def render_cluster() -> None:
    st.header("Level 4 - 군집 분석")
    _render_phase_banner()
    pivot, scaled, _region_lookup = _get_cluster_base()
    if scaled.empty:
        st.warning("군집 분석용 서울 시계열 데이터가 없습니다.")
        return

    n_clusters = st.slider("군집 수", min_value=2, max_value=6, value=4, key="level4_cluster_n")
    cluster_df = _get_clusters(int(n_clusters))
    st.plotly_chart(build_cluster_heatmap(scaled, cluster_df), width="stretch")
    st.plotly_chart(build_cluster_map(cluster_df), width="stretch")
    if not cluster_df.empty:
        st.dataframe(cluster_df[["district", "cluster_name"]].sort_values(["cluster_name", "district"]), width="stretch")


def render_anomaly() -> None:
    st.header("Level 4 - 이상거래")
    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    _render_phase_banner()
    region_name = st.selectbox("이상거래 탐지 지역", sorted(trade_df["_region_name"].dropna().unique()), index=0, key="level4_anomaly_region")
    region_code = str(trade_df.loc[trade_df["_region_name"] == region_name, "_lawd_cd"].iloc[0])
    year_options = sorted(int(value) for value in trade_df["year"].dropna().unique())
    default_years = tuple(year_options[-3:] if len(year_options) >= 3 else year_options)
    selected_years = tuple(st.multiselect("포함 연도", options=year_options, default=list(default_years), key="level4_anomaly_years"))
    if not selected_years:
        st.info("최소 1개 연도를 선택해주세요.")
        return

    anomaly_df = get_filtered_trade_anomalies(region_code, selected_years)
    st.plotly_chart(build_anomaly_chart(anomaly_df, region_name), width="stretch")
    if not anomaly_df.empty:
        st.metric("이상거래 비중", f"{anomaly_df['is_anomaly'].mean() * 100:.2f}%")


def render_did() -> None:
    st.header("Level 4 - 정책 효과(DiD)")
    _render_phase_banner()
    event_key = st.selectbox("이벤트 선택", list(DID_EVENTS.keys()), key="level4_did_event")
    did_df, did_result = _get_did(event_key)
    st.plotly_chart(build_parallel_trend_chart(did_df, DID_EVENTS[event_key]["event_ym"]), width="stretch")
    cols = st.columns(3)
    cols[0].metric("추정 효과", f"{did_result['pct_effect']:.2f}%" if did_result["pct_effect"] == did_result["pct_effect"] else "N/A")
    cols[1].metric("p-value", f"{did_result['did_pvalue']:.4f}" if did_result["did_pvalue"] == did_result["did_pvalue"] else "N/A")
    cols[2].metric("관측치 수", f"{did_result['n_obs']}")
    if st.toggle("회귀 요약 보기", value=False, key="level4_did_summary_toggle"):
        st.code(str(did_result["summary"]))


def render_cycle() -> None:
    st.header("Level 4 - 시장 사이클")
    cycle_df = load_cycle_features_precomputed()
    if cycle_df.empty:
        st.warning("선계산 시장 사이클 데이터가 없습니다.")
        return

    _render_phase_banner()
    use_hmm = st.toggle("HMM 사용(설치되어 있으면 자동 적용)", value=False, key="level4_cycle_hmm")
    st.plotly_chart(build_cycle_dashboard(cycle_df, use_hmm=use_hmm), width="stretch")