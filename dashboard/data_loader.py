"""대시보드용 데이터 로딩 유틸리티.

``@st.cache_data`` 를 사용하여 Streamlit 세션 간 데이터를 캐싱한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import PROCESSED_DIR, ALL_REGIONS, SEOUL_REGIONS, GYEONGGI_REGIONS


def _normalize_month_column(df: pd.DataFrame, ym_column: str = "ym") -> pd.DataFrame:
    """연월 컬럼을 문자열과 날짜형으로 정규화한다.

    Args:
        df: 정규화할 원본 DataFrame.
        ym_column: 연월 정보가 저장된 컬럼명.

    Returns:
        ``ym`` 문자열과 ``date`` 날짜형 컬럼이 보강된 DataFrame.
    """
    if ym_column not in df.columns:
        return df

    normalized_df = df.copy()
    normalized_df[ym_column] = (
        normalized_df[ym_column]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )
    normalized_df["date"] = pd.to_datetime(normalized_df[ym_column], format="%Y%m", errors="coerce")
    return normalized_df


@st.cache_data(ttl=3600)
def load_trade_summary() -> pd.DataFrame:
    """매매 월별 집계 데이터를 로드한다.

    Returns:
        매매 집계 DataFrame. 파일이 없으면 빈 DataFrame.
    """
    path = PROCESSED_DIR / "monthly_trade_summary.parquet"
    if not path.exists():
        return pd.DataFrame()
    return _normalize_month_column(pd.read_parquet(path))


@st.cache_data(ttl=3600)
def load_rent_summary() -> pd.DataFrame:
    """전월세 월별 집계 데이터를 로드한다.

    Returns:
        전월세 집계 DataFrame. 파일이 없으면 빈 DataFrame.
    """
    path = PROCESSED_DIR / "monthly_rent_summary.parquet"
    if not path.exists():
        return pd.DataFrame()
    return _normalize_month_column(pd.read_parquet(path))


@st.cache_data(ttl=3600)
def load_macro_monthly() -> pd.DataFrame:
    """거시지표 월별 통합 데이터를 로드한다.

    Returns:
        거시지표 DataFrame. 파일이 없으면 빈 DataFrame.
    """
    path = PROCESSED_DIR / "macro_monthly.parquet"
    if not path.exists():
        return pd.DataFrame()
    macro_df = pd.read_parquet(path)
    if "date" in macro_df.columns:
        macro_df = macro_df.copy()
        macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")
    return _normalize_month_column(macro_df)


def get_region_options() -> dict[str, str]:
    """사이드바 지역 선택에 사용할 옵션을 반환한다.

    Returns:
        ``{코드: 이름}`` 딕셔너리.
    """
    return ALL_REGIONS


def get_seoul_options() -> dict[str, str]:
    """서울 지역 옵션을 반환한다.

    Returns:
        ``{코드: 이름}`` 딕셔너리.
    """
    return SEOUL_REGIONS


def get_gyeonggi_options() -> dict[str, str]:
    """경기 지역 옵션을 반환한다.

    Returns:
        ``{코드: 이름}`` 딕셔너리.
    """
    return GYEONGGI_REGIONS
