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
    detect_anomalies,
    get_current_phase,
    load_anomaly_data,
    load_cycle_features,
    run_did_regression,
    run_dtw_clustering,
    run_prediction_model,
)
from dashboard.data_loader import load_trade_summary


@st.cache_data(ttl=3600)
def _get_prediction(scope_name: str):
    dataset = build_prediction_dataset(get_scope_codes(scope_name), scope_name)
    return run_prediction_model(dataset)


@st.cache_data(ttl=3600)
def _get_clusters(n_clusters: int):
    return run_dtw_clustering(n_clusters)


@st.cache_data(ttl=3600)
def _get_anomalies(region_code: str, years: tuple[int, ...]):
    raw_df = load_anomaly_data(region_code, list(years))
    return detect_anomalies(raw_df)


@st.cache_data(ttl=3600)
def _get_did(event_key: str):
    did_df = build_did_dataset(event_key)
    return did_df, run_did_regression(did_df)


@st.cache_data(ttl=3600)
def _get_cycle_features():
    return load_cycle_features()


def render() -> None:
    st.header("Level 4 - 고급 분석")

    trade_df = load_trade_summary()
    if trade_df.empty:
        st.warning("매매 집계 데이터가 없습니다.")
        return

    cycle_df = _get_cycle_features()
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

    scope_options = ["서울 전체", "경기 전체", "수도권 전체", *sorted(trade_df["_region_name"].unique())]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["가격 예측", "군집 분석", "이상거래", "정책 효과(DiD)", "시장 사이클"])

    with tab1:
        scope_name = st.selectbox("예측 범위", scope_options, index=0)
        pred_df, metrics, feature_cols = _get_prediction(scope_name)
        st.plotly_chart(build_prediction_chart(pred_df, scope_name), width="stretch")
        cols = st.columns(3)
        cols[0].metric("RMSE", f"{metrics['rmse']:,.0f}" if metrics["rmse"] == metrics["rmse"] else "N/A")
        cols[1].metric("MAPE", f"{metrics['mape']:.2f}%" if metrics["mape"] == metrics["mape"] else "N/A")
        cols[2].metric("R²", f"{metrics['r_squared']:.4f}" if metrics["r_squared"] == metrics["r_squared"] else "N/A")
        if feature_cols:
            st.caption("사용 특징량: " + ", ".join(feature_cols))

    with tab2:
        n_clusters = st.slider("군집 수", min_value=2, max_value=6, value=4)
        cluster_df = _get_clusters(int(n_clusters))
        st.plotly_chart(build_cluster_heatmap(), width="stretch")
        st.plotly_chart(build_cluster_map(cluster_df), width="stretch")
        if not cluster_df.empty:
            st.dataframe(cluster_df[["district", "cluster_name"]].sort_values(["cluster_name", "district"]), width="stretch")

    with tab3:
        region_name = st.selectbox("이상거래 탐지 지역", sorted(trade_df["_region_name"].unique()), index=0)
        region_code = trade_df.loc[trade_df["_region_name"] == region_name, "_lawd_cd"].iloc[0]
        year_options = sorted(int(value) for value in trade_df["year"].dropna().unique())
        selected_years = st.multiselect("포함 연도", options=year_options, default=year_options[-3:])
        if selected_years:
            anomaly_df = _get_anomalies(str(region_code), tuple(selected_years))
            st.plotly_chart(build_anomaly_chart(anomaly_df, region_name), width="stretch")
            st.metric("이상거래 비중", f"{anomaly_df['is_anomaly'].mean() * 100:.2f}%" if not anomaly_df.empty else "N/A")
        else:
            st.info("최소 1개 연도를 선택해주세요.")

    with tab4:
        event_key = st.selectbox("이벤트 선택", list(DID_EVENTS.keys()))
        did_df, did_result = _get_did(event_key)
        st.plotly_chart(build_parallel_trend_chart(did_df, DID_EVENTS[event_key]["event_ym"]), width="stretch")
        cols = st.columns(3)
        cols[0].metric("추정 효과", f"{did_result['pct_effect']:.2f}%" if did_result["pct_effect"] == did_result["pct_effect"] else "N/A")
        cols[1].metric("p-value", f"{did_result['did_pvalue']:.4f}" if did_result["did_pvalue"] == did_result["did_pvalue"] else "N/A")
        cols[2].metric("관측치 수", f"{did_result['n_obs']}")
        with st.expander("회귀 요약 보기"):
            st.code(str(did_result["summary"]))

    with tab5:
        use_hmm = st.toggle("HMM 사용(설치되어 있으면 자동 적용)", value=False)
        st.plotly_chart(build_cycle_dashboard(cycle_df, use_hmm=use_hmm), width="stretch")

