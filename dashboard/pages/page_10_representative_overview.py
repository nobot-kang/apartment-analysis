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
    st.header("Representative Level 1 - 대표단지 coverage")
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
    st.header("Representative Level 1 - 지역별 평당가 타임라인")
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
    st.header("Representative Level 1 - 지역별 59형 vs 84형 비교")
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
    st.header("Representative Level 1 - 특정 시점 지역 분포")
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
    st.header("Representative Level 1 - 단지별 pair gap 히스토리")
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
