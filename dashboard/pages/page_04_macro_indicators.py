"""Page 04 - Level 3 거시지표 연계 분석."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import get_scope_codes
from analysis.level3 import (
    build_correlation_heatmap,
    build_dual_correlation_heatmaps,
    build_fx_event_chart,
    build_m2_price_chart,
    build_macro_scatter,
    build_rate_lag_chart,
    build_real_price_chart,
    build_scope_frame,
    prepare_combined_correlation,
    prepare_fx_event_study,
    prepare_real_price_index,
)
from dashboard.data_loader import (
    get_scope_option_list,
    load_macro_monthly,
    load_rent_summary,
    load_trade_summary,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_scope_frame(scope_name: str):
    trade_df = load_trade_summary()
    rent_df = load_rent_summary()
    macro_df = load_macro_monthly()
    return build_scope_frame(trade_df, rent_df, macro_df, get_scope_codes(scope_name), scope_name)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_fx_event():
    trade_df = load_trade_summary()
    rent_df = load_rent_summary()
    macro_df = load_macro_monthly()
    combined = build_scope_frame(trade_df, rent_df, macro_df, get_scope_codes("수도권 전체"), "수도권 전체")
    return prepare_fx_event_study(combined)


def render_rate_lag() -> None:
    st.header("🌍 금리가 오르면 집값은 언제 내릴까?")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **핵심 개념 — 시차(Time Lag):** 금리가 오르면 집값이 바로 내리지 않습니다. 보통 몇 달이 지나야 효과가 나타납니다. 이 그래프는 '몇 달 뒤에 효과가 가장 크게 나타나는지'를 보여줍니다.

        **상관계수 막대 해석 (-1 ~ +1):**
        - **가장 큰 음수 막대 = 핵심 포인트**: 금리 인상 후 해당 개월 수가 지났을 때 집값 하락이 가장 강하게 나타남.
          - 예: 6개월 시차에서 −0.65 → 금리 올리고 6개월 후 집값 하락이 가장 뚜렷함
        - **양수 막대**: 금리가 오를수록 오히려 가격도 오르는 구간 (유동성 풍부 국면에서 가끔 발생)
        - **0 근처**: 해당 시차에서는 금리와 가격의 관계가 뚜렷하지 않음

        **⚠️ 주의:** 시차 효과는 경기 환경마다 달라집니다. 과거 패턴이 미래에도 똑같이 반복되리라는 보장은 없습니다.
        """)

    scope_name = st.selectbox("분석 범위", get_scope_option_list(), index=0, key="level3_rate_scope")
    combined = _get_scope_frame(scope_name)
    st.plotly_chart(build_rate_lag_chart(combined, scope_name), width="stretch")


def render_m2() -> None:
    st.header("🌍 돈이 많이 풀리면 집값이 오를까? (통화량)")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **M2(광의통화)란?** 시중에 유통되는 돈의 총량. 은행 예금, 적금 등 비교적 쉽게 현금화할 수 있는 돈을 모두 합한 것.

        **왜 집값과 관계가 있을까?** 돈이 많이 풀리면 사람들이 자산(주식, 부동산)을 사려는 수요가 늘어 가격이 오르는 경향이 있습니다.

        **그래프 해석:**
        - **두 선이 함께 오름**: 통화량 증가 → 집값도 상승 (인플레이션 국면에서 전형적인 패턴)
        - **통화량만 오르고 집값 횡보**: 돈이 풀려도 아직 부동산으로 유입되지 않는 시기
        - **통화량 증가율 (YoY %)**: 1년 전보다 시중 통화량이 몇 % 늘었는지. 10% 이상이면 상당히 빠른 증가.

        **💡 팁:** 통화량 증가가 집값에 영향을 주기까지 보통 6개월~1년의 시간이 걸립니다.
        """)

    scope_name = st.selectbox("분석 범위", get_scope_option_list(), index=0, key="level3_m2_scope")
    combined = _get_scope_frame(scope_name)
    st.plotly_chart(build_m2_price_chart(combined, scope_name), width="stretch")


def render_fx_event() -> None:
    st.header("🌍 환율 급변 때 아파트는?")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **이벤트 스터디란?** 특정 사건(환율 급등/급락) 전후로 집값이 어떻게 움직였는지 여러 사건을 평균 내서 보는 분석.

        **그래프 해석:**
        - **0 기준선 (이벤트 시점)**: 환율이 급변한 달.
        - **이벤트 이후 선이 위로**: 환율 급변 후 집값이 평균적으로 오른 패턴.
        - **이벤트 이후 선이 아래로**: 환율 급변 후 집값이 평균적으로 내린 패턴.
        - **신뢰구간 (음영 영역)**: 여러 이벤트 간의 편차. 음영이 넓으면 이벤트마다 반응이 달랐다는 뜻.

        **환율과 부동산의 관계:**
        - 환율 급등(원화 약세) → 외국인 투자자에게 한국 부동산이 싸 보임 → 수요 증가 가능
        - 환율 급등 → 수입 물가 상승 → 인플레이션 → 실물 자산 수요 증가 가능

        **⚠️ 주의:** 이 분석은 수도권 전체를 기준으로 계산됩니다.
        """)

    fx_df = _get_fx_event()
    st.plotly_chart(build_fx_event_chart(fx_df), width="stretch")


def render_real_price() -> None:
    st.header("🌍 물가 반영 후 실제 집값 변화")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **실질 가격이란?** 물가 상승분을 제거한 가격. '진짜 비싸진 건지, 돈의 가치가 낮아진 것뿐인지' 구분해줌.

        **예시:**
        - 2015년 집값 5억 → 2025년 집값 8억 (명목 가격 60% 상승)
        - 같은 기간 물가도 40% 올랐다면 → 실질 가격 상승은 14%에 불과

        **수치 해석 (지수 기준):**
        - **기준 시점 = 100**: 분석 시작 시점을 100으로 고정.
        - **지수 130**: 기준 시점 대비 물가를 뺀 실질 가격이 30% 오름.
        - **명목 가격 선 vs 실질 가격 선**: 두 선의 격차가 클수록 물가 상승이 집값 오름에 많이 기여한 것.

        **💡 팁:** 명목 가격은 올랐어도 실질 가격이 횡보하거나 내렸다면, '집값이 오른 게 아니라 화폐 가치가 떨어진 것'에 가깝습니다.
        """)

    scope_name = st.selectbox("분석 범위", get_scope_option_list(), index=0, key="level3_real_scope")
    combined = _get_scope_frame(scope_name)
    real_df = prepare_real_price_index(combined)
    st.plotly_chart(build_real_price_chart(real_df, scope_name), width="stretch")


def render_correlation() -> None:
    st.header("🌍 경제 지표들과 집값 — 상관관계 종합")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 히트맵으로 전체 관계 파악 → ② 두 지역 비교 → ③ 산점도로 특정 관계 확대

        **상관관계 히트맵 해석:**
        - **색 진한 빨강 (+0.7 이상)**: 두 지표가 같은 방향으로 강하게 움직임 (함께 오르거나 함께 내림)
        - **색 진한 파랑 (−0.7 이하)**: 두 지표가 반대 방향으로 강하게 움직임
        - **흰색 (0 근처)**: 두 지표 사이에 뚜렷한 관계 없음
        - **대각선**: 같은 지표끼리 비교 = 항상 1.0 (빨강)

        **R² (결정계수, 0 ~ 1) 해석 (산점도 하단):**
        - **R² = 0.7**: X축 지표가 변할 때 Y축 지표 변화의 70%를 설명할 수 있음.
        - **R² = 0.3 이하**: 두 지표 관계가 약함. 산점도의 점들이 흩어져 있을 것.
        - **R² = 1.0**: 완벽한 직선 관계 (실제로는 거의 불가능).

        **💡 팁:** 두 지역을 비교할 때 상관관계 패턴이 다르면, 두 지역이 다른 경제적 요인에 반응한다는 의미.
        """)

    scope_options = get_scope_option_list()
    scope_name = st.selectbox("기준 범위", scope_options, index=0, key="level3_corr_scope")
    combined = prepare_combined_correlation(_get_scope_frame(scope_name))
    if combined.empty:
        st.info("복합 상관관계 데이터가 없습니다.")
        return

    st.plotly_chart(build_correlation_heatmap(combined), width="stretch")

    compare_col1, compare_col2 = st.columns(2)
    scope_a = compare_col1.selectbox("비교 범위 A", scope_options, index=0, key="level3_corr_scope_a")
    scope_b = compare_col2.selectbox("비교 범위 B", scope_options, index=min(1, len(scope_options) - 1), key="level3_corr_scope_b")
    combined_a = prepare_combined_correlation(_get_scope_frame(scope_a))
    combined_b = prepare_combined_correlation(_get_scope_frame(scope_b))
    if not combined_a.empty and not combined_b.empty:
        st.plotly_chart(build_dual_correlation_heatmaps(combined_a, scope_a, combined_b, scope_b), width="stretch")

    numeric_cols = [column for column in combined.columns if column != "date"]
    x_col = st.selectbox("산점도 X축", numeric_cols, index=0, key="level3_scatter_x")
    y_candidates = [column for column in numeric_cols if column != x_col]
    y_col = st.selectbox("산점도 Y축", y_candidates, index=0, key="level3_scatter_y")
    fig_scatter, reg = build_macro_scatter(combined, x_col, y_col)
    st.plotly_chart(fig_scatter, width="stretch")
    st.metric("R²", f"{reg['r_squared']:.4f}" if reg["r_squared"] == reg["r_squared"] else "N/A")