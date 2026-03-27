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
    st.header("🔬 AI 가격 예측 모델")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 현재 시장 국면 배너 확인 → ② 예측 범위 선택 → ③ 그래프에서 예측 정확도 확인 → ④ 지표 카드 해석

        **그래프 해석:**
        - **파란 선 (실제 가격)**: 실제로 거래된 월별 평균 가격.
        - **빨간 선 (예측 가격)**: 모델이 예측한 가격. 두 선이 가까울수록 잘 맞은 것.
        - **선들이 많이 엇갈리는 구간**: 급격한 정책 변화나 외부 충격이 있던 시점.

        **예측 정확도 지표:**
        - **RMSE (만원)**: 예측이 평균적으로 몇 만원 틀렸는지. 낮을수록 좋음.
          - 예: RMSE 500만원 → 예측 가격이 실제와 평균 500만원 차이
        - **MAPE (%)**: 퍼센트로 나타낸 오차율. 5% 미만: 매우 정확. 10~20%: 방향성 참고 수준.
        - **R² (0~1)**: 모델이 가격 변동의 몇 %를 설명하는지. 0.8 이상이면 잘 맞는 모델.

        **⚠️ 주의:** 이 예측은 과거 데이터 기반입니다. 전쟁, 대규모 정책 변화 등 전례 없는 사건은 예측 불가합니다.
        """)

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
    st.header("🔬 비슷한 가격 패턴 지역 묶기")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **군집 분석이란?** 가격 변화 패턴이 비슷한 지역끼리 자동으로 묶어주는 분석. 마치 비슷한 학생들끼리 모둠을 나누는 것처럼.

        **그래프 해석:**
        - **히트맵**: 각 지역(행)의 연도별 가격 변화를 색으로 표현. 같은 군집(색깔)끼리 비슷한 패턴.
        - **지도**: 같은 색으로 칠해진 지역들이 한 군집. 지리적으로 멀어도 같은 색이면 가격 패턴이 유사.
        - **군집 수 슬라이더**: 2~6개 중 선택. 많을수록 세분화. 4개가 일반적으로 이해하기 쉬움.

        **군집 의미 예시:**
        - 강남권 군집: 고가 + 상승 선도
        - 외곽 군집: 중저가 + 강남권 따라가는 패턴
        - 특수 군집: 재개발·재건축 이슈 지역

        **💡 팁:** 군집이 다른 지역끼리 투자를 분산하면 리스크를 줄일 수 있습니다.
        """)

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
    st.header("🔬 수상한 거래 탐지")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이상거래란?** 통계적으로 평균에서 크게 벗어난 거래. 시세보다 훨씬 비싸거나 싸게 거래된 것들.

        **그래프 해석:**
        - **파란 점 (정상 거래)**: 해당 시기 시세에 맞게 거래된 건들.
        - **빨간 점 (이상 거래)**: 통계적으로 특이한 거래. 실제 이상거래일 수도, 오피스텔·특수 계약일 수도.
        - **이상거래 비중 (%)**: 전체 중 이상 거래로 분류된 비율. 5% 이상이면 해당 기간에 특이한 움직임이 많았다는 신호.

        **이상거래가 생기는 이유:**
        - 가족 간 증여성 거래 (시세보다 훨씬 싸게)
        - 투기성 고가 매수 (시세보다 훨씬 비싸게)
        - 데이터 오류

        **⚠️ 주의:** 이 탐지는 통계 모델 기반이므로, 빨간 점이 모두 실제 이상거래는 아닙니다. 참고용으로 활용하세요.
        """)

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
    st.header("🔬 정책 시행 전후 비교")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이중 차분법(DiD)이란?** 정책의 효과를 측정하는 통계 방법. '정책을 받은 지역'과 '받지 않은 지역'의 변화를 비교해 순수한 정책 효과만 분리합니다.

        **그래프 해석 (평행 추세 검증):**
        - **정책 시행 이전**: 두 그룹(처리 지역 vs 비교 지역)의 선이 비슷하게 움직여야 함 → "평행 추세 가정" 성립.
        - **정책 시행 이후**: 두 선이 벌어지는 정도 = 정책 효과.
        - **선들이 정책 전부터 이미 벌어져 있으면**: 비교 자체가 공평하지 않을 수 있음.

        **주요 수치 해석:**
        - **추정 효과 (%)**: 정책 후 처리 지역이 비교 지역보다 평균 얼마나 더/덜 변했는지.
          - 예: +8% → 정책 시행 지역이 비교 지역보다 평균 8% 더 오름
        - **p-value**: 이 효과가 우연일 확률.
          - 0.05 미만 → 95% 이상의 신뢰도로 의미 있는 효과
          - 0.05 이상 → 통계적으로 의미 있다고 보기 어려움
        - **관측치 수**: 분석에 사용된 데이터 건수. 많을수록 신뢰도 높음.
        """)

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
    st.header("🔬 지금 시장은 어느 단계일까?")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **시장 사이클이란?** 부동산 시장은 상승 → 과열 → 조정 → 침체를 반복하는 경향이 있습니다.

        **현재 국면 배너 해석:**
        - **기준금리 (%)**: 한국은행이 정한 기준 이자율. 높을수록 대출 부담 증가.
        - **M2 YoY (%)**: 1년 전보다 시중 통화량이 몇 % 늘었는지. 양수 = 돈 풀리는 중.
        - **거래량 MoM (%)**: 전월 대비 거래량 변화. 양수 = 거래 늘어나는 중.

        **사이클 4단계:**
        - **상승 국면**: 거래량↑ + 가격↑. 매수 심리 살아있음.
        - **과열 국면**: 거래량 정체 + 가격 급등. 고점 가능성 주의.
        - **조정 국면**: 거래량↓ + 가격 하락 시작. 관망세 증가.
        - **침체 국면**: 거래량↓↓ + 가격↓↓. 급매물 출현.

        **💡 팁:** HMM(숨은마르코프모델) 옵션을 켜면 통계 모델로 국면을 자동 분류합니다. 꺼두면 규칙 기반 분류.
        """)

    cycle_df = load_cycle_features_precomputed()
    if cycle_df.empty:
        st.warning("선계산 시장 사이클 데이터가 없습니다.")
        return

    _render_phase_banner()
    use_hmm = st.toggle("HMM 사용(설치되어 있으면 자동 적용)", value=False, key="level4_cycle_hmm")
    st.plotly_chart(build_cycle_dashboard(cycle_df, use_hmm=use_hmm), width="stretch")