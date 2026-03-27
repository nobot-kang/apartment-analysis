"""Dashboard page for representative 59/84 complex overview analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.representative_complex_analysis import (
    build_band_comparison_chart,
    build_band_comparison_frame,
    build_pair_gap_history_chart,
    build_pair_gap_history_frame,
    build_region_timeline_chart,
    build_region_timeline_frame,
    build_representative_coverage_chart,
    build_representative_coverage_frame,
    build_snapshot_distribution_chart,
    build_snapshot_distribution_frame,
    get_region_option_map,
    list_complex_options,
)
from dashboard.data_loader import (
    load_representative_complex_universe,
    load_representative_pair_gap_monthly,
    load_representative_region_monthly,
    load_representative_trade_band_monthly,
)


def _format_band(value: int) -> str:
    return f"{int(value)}형"


def _select_region(region_options: dict[str, str], key: str, label: str = "지역") -> str | None:
    if not region_options:
        return None
    labels = list(region_options.keys())
    selected = st.selectbox(label, labels, index=0, key=key)
    return region_options[selected]


@st.cache_data(ttl=3600, show_spinner=False)
def _get_coverage(region_level: str) -> tuple[pd.DataFrame, dict[str, int]]:
    universe_df = load_representative_complex_universe()
    coverage_df = build_representative_coverage_frame(universe_df, region_level=region_level)
    metrics = {
        "complex_count": int(universe_df["aptSeq"].nunique()) if not universe_df.empty else 0,
        "pair_count": int(universe_df.get("is_pair_complex", pd.Series(dtype=bool)).fillna(False).sum()) if not universe_df.empty else 0,
        "count_59": int(universe_df.get("has_59_any", pd.Series(dtype=bool)).fillna(False).sum()) if not universe_df.empty else 0,
        "count_84": int(universe_df.get("has_84_any", pd.Series(dtype=bool)).fillna(False).sum()) if not universe_df.empty else 0,
    }
    return coverage_df, metrics


@st.cache_data(ttl=3600, show_spinner=False)
def _get_region_options(region_level: str) -> dict[str, str]:
    return get_region_option_map(load_representative_region_monthly(), region_level)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_region_timeline(region_level: str, market_type: str, area_band: int, region_key: str) -> pd.DataFrame:
    return build_region_timeline_frame(
        load_representative_region_monthly(),
        region_level=region_level,
        market_type=market_type,
        area_band=area_band,
        region_key=region_key,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_band_comparison(region_level: str, market_type: str, region_key: str) -> pd.DataFrame:
    return build_band_comparison_frame(
        load_representative_region_monthly(),
        region_level=region_level,
        market_type=market_type,
        region_key=region_key,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_distribution_snapshot(ym: str, region_level: str, area_band: int) -> pd.DataFrame:
    return build_snapshot_distribution_frame(
        load_representative_trade_band_monthly(),
        ym=ym,
        region_level=region_level,
        area_band=area_band,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_pair_complex_options(region_level: str, region_key: str | None) -> pd.DataFrame:
    universe_df = load_representative_complex_universe()
    pair_universe = universe_df[universe_df.get("is_pair_complex", False).fillna(False)].copy()
    return list_complex_options(pair_universe, region_level=region_level, region_key=region_key)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_pair_gap_history(apt_seq: str) -> pd.DataFrame:
    return build_pair_gap_history_frame(load_representative_pair_gap_monthly(), apt_seq=apt_seq)


def render_representative_coverage() -> None:
    st.header("🏡 대표 단지 데이터 커버리지")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **대표 단지란?** 서울 각 지역에서 59㎡(약 18평)와 84㎡(약 25평) 평형이 모두 있어 비교 가능한 아파트 단지들.

        **주요 수치 해석:**
        - **대표단지 수**: 분석에 포함된 단지 수.
        - **Pair 단지 수**: 59형·84형이 모두 있어 두 평형을 직접 비교할 수 있는 단지 수.
        - **59형 보유 / 84형 보유**: 각 평형이 있는 단지 수.

        **커버리지 그래프 해석:**
        - 막대가 짧거나 공백이 많은 지역 = 그 지역의 데이터가 적어 분석 결과가 불안정할 수 있음.
        - 전체 커버리지가 높을수록 분석의 신뢰도가 높아짐.

        **💡 팁:** 지역 단위를 '시군구'에서 '법정동'으로 바꾸면 더 세밀하게 볼 수 있지만 데이터가 분산됩니다.
        """)

    region_level = st.radio(
        "지역 단위",
        options=["sigungu", "bjdong"],
        horizontal=True,
        format_func=lambda value: "시군구" if value == "sigungu" else "법정동",
        key="rep_cov_region_level",
    )
    coverage_df, metrics = _get_coverage(region_level)
    if coverage_df.empty:
        st.warning("대표단지 coverage 데이터가 없습니다.")
        return

    cols = st.columns(4)
    cols[0].metric("대표단지 수", f"{metrics['complex_count']:,}")
    cols[1].metric("Pair 단지 수", f"{metrics['pair_count']:,}")
    cols[2].metric("59형 보유", f"{metrics['count_59']:,}")
    cols[3].metric("84형 보유", f"{metrics['count_84']:,}")
    st.plotly_chart(build_representative_coverage_chart(coverage_df), width="stretch")
    st.dataframe(coverage_df, width="stretch", hide_index=True)


def render_representative_region_timeline() -> None:
    st.header("🏡 지역별 평당가 흐름")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **평당가란?** 1평(약 3.3㎡)당 가격(만원). 서울 아파트는 보통 평당 3,000~1억원 이상 범위.

        **평당가로 보는 이유:** 면적이 다른 아파트들을 공평하게 비교하기 위해 면적을 통일한 지표.
        - 예: 서울 강남구 평당가 1억 = 25평(84㎡) 아파트 기준 25억

        **주요 수치 해석:**
        - **최근 대표 평당가 (만원/평)**: 가장 최근 달의 해당 지역 중앙값 평당가.
        - **활성 단지 수**: 해당 월에 데이터가 있는 단지 수(실거래 + 보간 포함).
        - **실관측 단지 수**: 실제 거래가 있어서 직접 계산된 단지 수 (보간 제외). 활성 단지 대비 비율이 낮으면 그 달 실거래가 드물었다는 의미.

        **선 그래프 해석:** 선이 오르면 그 지역·평형의 평당가가 상승 중.
        """)

    controls = st.columns([1.0, 1.0, 1.0, 1.5])
    region_level = controls[0].radio(
        "지역 단위",
        options=["sigungu", "bjdong"],
        horizontal=True,
        format_func=lambda value: "시군구" if value == "sigungu" else "법정동",
        key="rep_timeline_region_level",
    )
    market_type = controls[1].radio(
        "시장",
        options=["sale", "jeonse", "wolse"],
        horizontal=True,
        format_func=lambda value: {"sale": "매매", "jeonse": "전세", "wolse": "월세"}[value],
        key="rep_timeline_market",
    )
    area_band = int(
        controls[2].radio(
            "대표 평형",
            options=[59, 84],
            horizontal=True,
            format_func=_format_band,
            key="rep_timeline_band",
        )
    )
    region_key = _select_region(_get_region_options(region_level), "rep_timeline_region_key")
    if region_key is None:
        st.warning("선택할 수 있는 지역이 없습니다.")
        return

    trend_df = _get_region_timeline(region_level, market_type, area_band, region_key)
    if trend_df.empty:
        st.warning("선택한 조건의 지역 데이터가 없습니다.")
        return

    latest = trend_df.sort_values("date").iloc[-1]
    cols = st.columns(3)
    cols[0].metric("최근 대표 평당가", f"{latest['complex_eq_median_py']:,.1f} 만원/평")
    cols[1].metric("활성 단지 수", f"{int(latest['complex_count_active']):,}")
    cols[2].metric("실관측 단지 수", f"{int(latest['complex_count_observed']):,}")
    title = f"{latest['region_name']} · {_format_band(area_band)} · {'매매' if market_type == 'sale' else '전세' if market_type == 'jeonse' else '월세'}"
    st.plotly_chart(build_region_timeline_chart(trend_df, title), width="stretch")
    st.dataframe(trend_df.tail(24).sort_values("date", ascending=False), width="stretch", hide_index=True)


def render_representative_band_comparison() -> None:
    st.header("🏡 59형 vs 84형 가격 비교")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **59형 vs 84형이란?** 전용면적 59㎡(약 18평, 방 2~3개)와 84㎡(약 25평, 방 3개)를 비교.

        **주요 수치 해석:**
        - **59형 평당가 / 84형 평당가**: 각 평형의 1평당 가격.
        - **84/59 격차율 (%)**: 84형이 59형보다 몇 % 더 비싼지.
          - 예: 격차율 20% → 59형 5,000만원/평이면 84형은 6,000만원/평
          - 격차율이 0%에 가까우면 두 평형이 비슷한 가격
          - 격차율이 마이너스이면 오히려 59형이 더 비싼 역전 현상

        **그래프에서 두 선이 벌어지면:** 84형이 59형보다 더 빠르게 오르는 중 (대형 선호 강해짐).
        **두 선이 좁혀지면:** 59형이 상대적으로 강세 (소형 선호 강해지거나 대형 약세).
        """)

    controls = st.columns([1.0, 1.0, 1.5])
    region_level = controls[0].radio(
        "지역 단위",
        options=["sigungu", "bjdong"],
        horizontal=True,
        format_func=lambda value: "시군구" if value == "sigungu" else "법정동",
        key="rep_compare_region_level",
    )
    market_type = controls[1].radio(
        "시장",
        options=["sale", "jeonse", "wolse"],
        horizontal=True,
        format_func=lambda value: {"sale": "매매", "jeonse": "전세", "wolse": "월세"}[value],
        key="rep_compare_market",
    )
    region_key = _select_region(_get_region_options(region_level), "rep_compare_region_key")
    if region_key is None:
        st.warning("선택할 수 있는 지역이 없습니다.")
        return

    comparison_df = _get_band_comparison(region_level, market_type, region_key)
    if comparison_df.empty:
        st.warning("선택한 지역의 59형/84형 비교 데이터가 없습니다.")
        return

    latest = comparison_df.sort_values("date").iloc[-1]
    cols = st.columns(3)
    cols[0].metric("59형", f"{latest.get('complex_eq_median_py_59', float('nan')):,.1f} 만원/평")
    cols[1].metric("84형", f"{latest.get('complex_eq_median_py_84', float('nan')):,.1f} 만원/평")
    cols[2].metric("84/59 격차율", f"{latest.get('gap_ratio', float('nan')):,.2f}%")
    title = f"{latest['region_name']} · {'매매' if market_type == 'sale' else '전세' if market_type == 'jeonse' else '월세'}"
    st.plotly_chart(build_band_comparison_chart(comparison_df, title), width="stretch")
    st.dataframe(comparison_df.tail(24).sort_values("date", ascending=False), width="stretch", hide_index=True)


def render_representative_distribution_snapshot() -> None:
    st.header("🏡 특정 시점 지역별 가격 분포")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **보는 순서:** ① 평형·지역 단위·기준 월 선택 → ② 분포 그래프 확인 → ③ 단지별 표 확인

        **그래프 해석:**
        - **각 막대**: 지역별 평균 평당가. 막대가 높을수록 비싼 지역.
        - **가로축 지역 순서**: 평당가 높은 순으로 정렬.
        - **기준 월 바꾸기**: 과거와 현재를 비교해 어느 지역이 많이 올랐는지 확인 가능.

        **표의 'is_trade_imputed' 열:**
        - **False**: 해당 달에 실제 거래가 있어 직접 계산된 가격.
        - **True**: 해당 달에 거래가 없어 앞뒤 데이터로 보간(추정)된 가격 → 신뢰도가 낮을 수 있음.

        **💡 팁:** '보간 사용' 단지가 많은 지역은 그 달의 가격이 부정확할 수 있으므로 주의하세요.
        """)

    trade_df = load_representative_trade_band_monthly()
    if trade_df.empty:
        st.warning("대표단지 거래 band 데이터가 없습니다.")
        return

    ym_options = sorted(trade_df["ym"].dropna().astype(str).unique())
    controls = st.columns([1.0, 1.0, 1.5])
    area_band = int(
        controls[0].radio(
            "대표 평형",
            options=[59, 84],
            horizontal=True,
            format_func=_format_band,
            key="rep_dist_band",
        )
    )
    region_level = controls[1].radio(
        "지역 단위",
        options=["sigungu", "bjdong"],
        horizontal=True,
        format_func=lambda value: "시군구" if value == "sigungu" else "법정동",
        key="rep_dist_region_level",
    )
    ym = controls[2].selectbox("기준 월", ym_options, index=max(len(ym_options) - 1, 0), key="rep_dist_ym")

    distribution_df = _get_distribution_snapshot(ym, region_level, area_band)
    if distribution_df.empty:
        st.warning("선택한 조건의 분포 데이터가 없습니다.")
        return

    st.caption("단지별 대표 평당가를 먼저 만들고 지역 분포를 집계합니다.")
    st.plotly_chart(
        build_snapshot_distribution_chart(distribution_df, f"{ym} · {_format_band(area_band)} 분포"),
        width="stretch",
    )
    st.dataframe(
        distribution_df[["region_label", "apt_name_repr", "price_per_py_filled", "trade_count_obs", "is_trade_imputed"]]
        .sort_values(["region_label", "price_per_py_filled"], ascending=[True, False]),
        width="stretch",
        hide_index=True,
    )


def render_representative_pair_gap_history() -> None:
    st.header("🏡 단지별 59형·84형 가격 차이 추이")
    with st.expander("📖 이 페이지 읽는 법"):
        st.markdown("""
        **Pair Gap이란?** 같은 단지 안에서 84형과 59형의 평당가 차이 비율. 같은 단지이므로 입지는 동일, 순수하게 '평형 크기의 가치'만 비교.

        **주요 수치 해석:**
        - **84/59 격차율 (%)**: 84형이 59형보다 평당 몇 % 더 비싼지.
          - 예: 격차율 15% → 59형 5,000만원/평, 84형 5,750만원/평
          - 격차율 0% → 두 평형이 같은 평당가 (면적이 클수록 총액은 비싸지만 평당은 같음)
          - 격차율 마이너스 → 59형이 84형보다 평당 더 비쌈 (소형 희소성 또는 재건축 이슈)
        - **보간 사용**: "예"이면 해당 달 실거래 없이 추정값 사용 → 신뢰도 주의.
        - **fill age (표)**: 보간에 사용된 앞뒤 데이터가 몇 달 전/후 것인지. 클수록 오래된 데이터로 채워진 것.

        **선 그래프 위로 올라가면:** 84형과 59형 격차 벌어지는 중 (대형 선호 강해짐).
        """)

    controls = st.columns([1.0, 1.2, 1.8])
    region_level = controls[0].radio(
        "지역 단위",
        options=["sigungu", "bjdong"],
        horizontal=True,
        format_func=lambda value: "시군구" if value == "sigungu" else "법정동",
        key="rep_gap_region_level",
    )
    region_key = _select_region(_get_region_options(region_level), "rep_gap_region_key")
    if region_key is None:
        st.warning("선택할 수 있는 지역이 없습니다.")
        return

    complex_options = _get_pair_complex_options(region_level, region_key)
    if complex_options.empty:
        st.warning("선택한 지역에 pair 단지가 없습니다.")
        return

    complex_map = {str(row["label"]): str(row["aptSeq"]) for _, row in complex_options.iterrows()}
    selected_label = controls[2].selectbox("단지", list(complex_map.keys()), key="rep_gap_complex")
    gap_df = _get_pair_gap_history(complex_map[selected_label])
    if gap_df.empty:
        st.warning("선택한 단지의 pair gap 데이터가 없습니다.")
        return

    latest = gap_df.sort_values("date").iloc[-1]
    cols = st.columns(4)
    cols[0].metric("59형 평당가", f"{latest.get('sale_py_59_roll3', float('nan')):,.1f} 만원/평")
    cols[1].metric("84형 평당가", f"{latest.get('sale_py_84_roll3', float('nan')):,.1f} 만원/평")
    cols[2].metric("84/59 격차율", f"{latest.get('sale_gap_ratio', float('nan')):,.2f}%")
    cols[3].metric("보간 사용", "예" if bool(latest.get("sale_any_imputed", False)) else "아니오")
    st.plotly_chart(build_pair_gap_history_chart(gap_df, f"{selected_label} · 59/84 pair gap"), width="stretch")
    st.dataframe(
        gap_df[
            [
                "ym",
                "sale_py_59_roll3",
                "sale_py_84_roll3",
                "sale_gap_ratio",
                "sale_59_fill_age",
                "sale_84_fill_age",
                "sale_any_imputed",
            ]
        ].sort_values("ym", ascending=False),
        width="stretch",
        hide_index=True,
    )
