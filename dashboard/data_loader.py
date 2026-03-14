"""대시보드용 데이터 로딩 유틸리티."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.common import load_macro_monthly_df, load_rent_detail_df, load_rent_summary_df, load_trade_detail_df, load_trade_summary_df
from config.settings import ALL_REGIONS, GYEONGGI_REGIONS, SEOUL_REGIONS


@st.cache_data(ttl=3600)
def load_processed_data(data_type: str = "trade") -> pd.DataFrame:
    """전처리된 상세 데이터를 로드한다."""
    if data_type == "trade":
        return load_trade_detail_df()
    return load_rent_detail_df()


@st.cache_data(ttl=3600)
def load_trade_summary() -> pd.DataFrame:
    """정규화된 매매 월별 집계를 로드한다."""
    return load_trade_summary_df()


@st.cache_data(ttl=3600)
def load_rent_summary() -> pd.DataFrame:
    """정규화된 전월세 월별 집계를 로드한다."""
    return load_rent_summary_df()


@st.cache_data(ttl=3600)
def load_macro_monthly() -> pd.DataFrame:
    """월별 거시지표 통합 데이터를 로드한다."""
    return load_macro_monthly_df()


def get_region_options() -> dict[str, str]:
    """전체 지역 옵션을 반환한다."""
    return ALL_REGIONS


def get_seoul_options() -> dict[str, str]:
    """서울 지역 옵션을 반환한다."""
    return SEOUL_REGIONS


def get_gyeonggi_options() -> dict[str, str]:
    """경기 지역 옵션을 반환한다."""
    return GYEONGGI_REGIONS
