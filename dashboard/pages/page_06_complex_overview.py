"""Dashboard page for complex-level descriptive analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.complex_analysis import (
    build_complex_profile_chart,
    build_complex_profile_frame,
    build_coverage_report,
    build_density_heatmap,
    build_density_matrix,
    build_land_premium_chart,
    build_latest_snapshot,
    build_parking_premium_chart,
    build_parking_premium_frame,
    build_scale_premium_chart,
    build_scale_premium_frame,
)
from dashboard.data_loader import load_complex_master, load_complex_monthly_panel


@st.cache_data(ttl=3600, show_spinner=False)
def _get_master_diagnostics():
    master_df = load_complex_master()
    return build_complex_profile_frame(master_df), build_coverage_report(master_df)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_snapshot(months: int):
    return build_latest_snapshot(load_complex_monthly_panel(), months=months)


def _parking_coverage_pct() -> float:
    coverage_df = build_coverage_report(load_complex_master())
    row = coverage_df[coverage_df["feature"] == "세대당 주차대수"]
    if row.empty:
        return 0.0
    return float(row["coverage_pct"].iloc[0])


def render_complex_profile() -> None:
    st.header("🏘️ 단지 규모·특성 프로필")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 요약 카드(데이터 규모 확인) → ② 분포 그래프 → ③ 데이터 커버리지 표

        **주요 수치 해석:**
        - **단지 수**: 분석에 포함된 아파트 단지 수.
        - **법정동 수**: 데이터가 커버하는 행정 구역 수.
        - **중위 세대수**: 전체 단지 중 딱 가운데 단지의 세대수. 평균보다 극단값에 덜 영향 받음.
        - **주차 커버리지 (%)**: 주차 정보가 있는 단지 비율. 낮으면 주차 분석이 부정확할 수 있음.
        - **커버리지 표**: 각 특성(면적, 세대수 등)의 데이터가 얼마나 채워져 있는지 퍼센트로 표시.

        **💡 팁:** 커버리지가 낮은 특성은 분석 결과를 신중하게 해석해야 합니다.
        """)

    master_df = load_complex_master()
    if master_df.empty:
        st.warning("complex_master.parquet 가 없습니다. 집계 파이프라인을 먼저 실행해 주세요.")
        return

    profile_df, coverage_df = _get_master_diagnostics()
    cols = st.columns(4)
    cols[0].metric("단지 수", f"{master_df['aptSeq'].nunique():,}")
    cols[1].metric("법정동 수", f"{master_df['dong_repr'].nunique():,}" if "dong_repr" in master_df.columns else "N/A")
    cols[2].metric("중위 세대수", f"{master_df['household_count'].median():,.0f}" if "household_count" in master_df.columns else "N/A")
    parking_cov = coverage_df.loc[coverage_df["feature"] == "세대당 주차대수", "coverage_pct"]
    cols[3].metric("주차 정보 커버리지", f"{parking_cov.iloc[0]:.1f}%" if not parking_cov.empty else "0.0%")
    st.plotly_chart(build_complex_profile_chart(profile_df), width="stretch")
    st.dataframe(coverage_df, width="stretch", hide_index=True)


def render_complex_scale_premium() -> None:
    st.header("🏘️ 대단지일수록 얼마나 비쌀까?")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 집계 기간 선택 → ② 세대수 구간별 가격 차이 확인

        **주요 수치 해석:**
        - **세대수 구간별 평균 가격**: 단지 규모가 클수록 가격이 높은지 확인.
        - **대단지 프리미엄**: 세대수가 많은 단지가 평균보다 얼마나 더 비싸게 거래되는지.
          - 예: 1,000세대 이상 구간이 +10% → 대단지 아파트가 평균보다 10% 비쌈
        - **막대 높낮이**: 높을수록 해당 규모 단지의 평균 가격이 높음.

        **대단지가 비싼 이유:**
        - 단지 내 커뮤니티 시설(헬스장, 수영장 등) 규모
        - 관리비 효율화
        - 브랜드 단지 여부와 겹치는 경향

        **⚠️ 주의:** 규모 자체보다 입지(강남 vs 외곽)의 영향이 더 클 수 있어, 단순 비교는 조심해야 합니다.
        """)

    months = int(st.select_slider("집계 기간", options=[6, 12, 24], value=12, key="complex_scale_months"))
    snapshot_df = _get_snapshot(months)
    if snapshot_df.empty:
        st.warning("최근 단지 스냅샷 데이터가 없습니다.")
        return

    scale_df = build_scale_premium_frame(snapshot_df)
    st.plotly_chart(build_scale_premium_chart(scale_df), width="stretch")
    st.dataframe(scale_df, width="stretch", hide_index=True)


def render_complex_parking_premium() -> None:
    st.header("🏘️ 주차 공간이 가격에 미치는 영향")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 집계 기간 선택 → ② 세대당 주차 대수 구간별 가격 확인

        **주요 수치 해석:**
        - **세대당 주차 대수**: 단지 전체 주차 공간 ÷ 세대수. 1.0이면 세대당 주차 1대 가능.
        - **주차 프리미엄**: 주차가 넉넉한 단지가 평균보다 얼마나 더 비싼지.
          - 예: 세대당 1.5대 이상 구간 +8% → 넉넉한 주차 단지가 평균보다 8% 비쌈
        - **상관계수 (−1 ~ +1)**: 주차 대수와 가격이 얼마나 같이 움직이는지.
          - +0.3 이상이면 의미 있는 양의 관계. 0에 가까우면 주차가 가격에 별 영향 없음.

        **⚠️ 주의:** 주차 정보 커버리지가 낮은 경우 이 분석의 신뢰도가 떨어질 수 있습니다.
        """)

    coverage_pct = _parking_coverage_pct()
    if coverage_pct < 5:
        st.warning("현재 원천 데이터에 총주차수 컬럼이 거의 없어 주차 분석은 제한적으로만 표시됩니다.")

    months = int(st.select_slider("집계 기간", options=[6, 12, 24], value=12, key="complex_parking_months"))
    snapshot_df = _get_snapshot(months)
    parking_df = build_parking_premium_frame(snapshot_df)
    if parking_df.empty:
        st.info("주차 정보가 충분한 단지가 아직 없습니다.")
        return

    st.plotly_chart(build_parking_premium_chart(parking_df), width="stretch")
    st.dataframe(parking_df, width="stretch", hide_index=True)


def render_complex_density_premium() -> None:
    st.header("🏘️ 건물 밀도와 가격의 관계")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **용어 설명:**
        - **용적률**: 땅 면적 대비 건물 총 연면적 비율. 높을수록 높은 건물이 빽빽하게 들어선 것. 예: 용적률 300% = 100평 땅에 300평짜리 건물.
        - **건폐율**: 땅 면적 대비 1층 바닥 면적 비율. 낮을수록 땅 위에 여유 공간이 많음. 예: 건폐율 20% = 땅의 80%가 마당·조경.

        **히트맵 해석:**
        - **행(세로축)**: 용적률 구간 (높을수록 고층·고밀)
        - **열(가로축)**: 건폐율 구간 (낮을수록 개방감 있음)
        - **색 진한 빨강**: 해당 밀도 조합에서 가격이 가장 높은 구간
        - **일반적 패턴**: 용적률 낮고(저층) + 건폐율 낮은(개방적) 단지가 고급 단지 경향

        **💡 팁:** 매매/전세/월세 지표를 바꿔가며 각 시장에서 선호하는 밀도가 다른지 비교해보세요.
        """)

    months = int(st.select_slider("집계 기간", options=[6, 12, 24], value=12, key="complex_density_months"))
    snapshot_df = _get_snapshot(months)
    metric = st.radio(
        "가격 지표",
        options=["trade_price_per_m2", "jeonse_deposit_per_m2", "wolse_monthly_rent_per_m2"],
        format_func=lambda value: {
            "trade_price_per_m2": "매매 가격",
            "jeonse_deposit_per_m2": "전세 가격",
            "wolse_monthly_rent_per_m2": "월세 가격",
        }[value],
        horizontal=True,
        key="complex_density_metric",
    )
    matrix_df = build_density_matrix(snapshot_df, value_col=metric)
    st.plotly_chart(build_density_heatmap(matrix_df, "밀도 조합별 가격 매트릭스"), width="stretch")
    if not matrix_df.empty:
        st.dataframe(matrix_df, width="stretch")


def render_complex_land_premium() -> None:
    st.header("🏘️ 세대당 땅 넓이와 가격의 관계")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **세대당 대지면적이란?** 단지 전체 땅 면적을 세대수로 나눈 값. 각 세대가 '이론적으로' 소유한 땅의 넓이.

        **왜 중요할까?** 재건축 시 각 세대가 받는 땅 지분이 크면 재건축 후 더 큰 평수를 받을 수 있음 → 재건축 기대가격에 영향.

        **그래프 해석:**
        - **산점도 각 점**: 단지 하나. X축 = 세대당 대지면적, Y축 = 가격.
        - **우상향 추세**: 땅이 넓을수록 비싼 경향.
        - **상관계수**: 두 값이 얼마나 같이 움직이는지.
          - +0.5 이상이면 강한 양의 관계 (땅 넓을수록 확실히 비쌈)
          - 0에 가까우면 땅 넓이가 가격에 별로 영향 없음

        **💡 팁:** 오래된 저층 단지는 세대당 대지면적이 크지만 가격이 낮은 경우도 있습니다. 입지와 함께 봐야 해요.
        """)

    months = int(st.select_slider("집계 기간", options=[6, 12, 24], value=12, key="complex_land_months"))
    snapshot_df = _get_snapshot(months)
    if snapshot_df.empty:
        st.warning("최근 단지 스냅샷 데이터가 없습니다.")
        return

    st.plotly_chart(build_land_premium_chart(snapshot_df), width="stretch")
    valid = snapshot_df[["avg_land_area_per_household", "trade_price_per_m2"]].dropna()
    if not valid.empty:
        st.metric("상관계수", f"{valid['avg_land_area_per_household'].corr(valid['trade_price_per_m2']):.3f}")
