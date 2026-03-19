"""대시보드용 데이터 로딩 유틸리티."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import (
    get_scope_options,
    load_complex_forecast_targets_df,
    load_complex_master_df,
    load_complex_monthly_panel_df,
    load_dashboard_conversion_rate_df,
    load_dashboard_cycle_features_df,
    load_dashboard_district_year_metrics_df,
    load_dashboard_jeonse_ratio_df,
    load_dashboard_trade_anomalies_df,
    load_macro_monthly_df,
    load_rent_detail_df,
    load_rent_summary_df,
    load_trade_detail_df,
    load_trade_summary_df,
)
from config.settings import ALL_REGIONS, GYEONGGI_REGIONS, SEOUL_REGIONS


@st.cache_resource(show_spinner=False)
def load_processed_data(data_type: str = "trade") -> pd.DataFrame:
    """전처리된 상세 데이터를 로드한다."""
    if data_type == "trade":
        return load_trade_detail_df()
    return load_rent_detail_df()


@st.cache_resource(show_spinner=False)
def load_trade_summary() -> pd.DataFrame:
    """정규화된 매매 월별 집계를 로드한다."""
    return load_trade_summary_df()


@st.cache_resource(show_spinner=False)
def load_rent_summary() -> pd.DataFrame:
    """정규화된 전월세 월별 집계를 로드한다."""
    return load_rent_summary_df()


@st.cache_resource(show_spinner=False)
def load_macro_monthly() -> pd.DataFrame:
    """월별 거시지표 통합 데이터를 로드한다."""
    return load_macro_monthly_df()


@st.cache_resource(show_spinner=False)
def load_complex_master() -> pd.DataFrame:
    """단지 정적 특성 마스터를 로드한다."""
    return load_complex_master_df()


@st.cache_resource(show_spinner=False)
def load_complex_monthly_panel() -> pd.DataFrame:
    """단지-월 패널 데이터를 로드한다."""
    return load_complex_monthly_panel_df()


@st.cache_resource(show_spinner=False)
def load_complex_forecast_targets() -> pd.DataFrame:
    """단지 예측용 타깃 패널을 로드한다."""
    return load_complex_forecast_targets_df()


@st.cache_resource(show_spinner=False)
def load_jeonse_ratio_precomputed() -> pd.DataFrame:
    """선계산된 전세가율 데이터를 로드한다."""
    return load_dashboard_jeonse_ratio_df()


@st.cache_resource(show_spinner=False)
def load_conversion_rate_precomputed() -> pd.DataFrame:
    """선계산된 전월세 전환율 데이터를 로드한다."""
    return load_dashboard_conversion_rate_df()


@st.cache_resource(show_spinner=False)
def load_district_year_metrics() -> pd.DataFrame:
    """선계산된 연도별 지역 지표를 로드한다."""
    return load_dashboard_district_year_metrics_df()


@st.cache_resource(show_spinner=False)
def load_cycle_features_precomputed() -> pd.DataFrame:
    """선계산된 시장 사이클 특징량을 로드한다."""
    return load_dashboard_cycle_features_df()


@st.cache_resource(show_spinner=False)
def load_trade_anomalies_precomputed() -> pd.DataFrame:
    """선계산된 이상거래 데이터를 로드한다."""
    return load_dashboard_trade_anomalies_df()


@st.cache_data(ttl=3600, show_spinner=False)
def get_filtered_trade_anomalies(region_code: str, years: tuple[int, ...]) -> pd.DataFrame:
    """선계산 이상거래 데이터를 지역/연도로 필터링한다."""
    return load_dashboard_trade_anomalies_df(
        years=list(years),
        region_codes=[region_code],
        columns=["date", "year", "region_code", "price_per_m2", "is_anomaly", "apt_name_repr"],
    )


@st.cache_resource(show_spinner=False)
def get_scope_option_list() -> list[str]:
    """대시보드 표준 scope 옵션 목록을 반환한다."""
    return get_scope_options()


def get_region_options() -> dict[str, str]:
    """전체 지역 옵션을 반환한다."""
    return ALL_REGIONS


def get_seoul_options() -> dict[str, str]:
    """서울 지역 옵션을 반환한다."""
    return SEOUL_REGIONS


def get_gyeonggi_options() -> dict[str, str]:
    """경기 지역 옵션을 반환한다."""
    return GYEONGGI_REGIONS
