"""공통 전처리: region / area_bucket / month 컬럼 파생."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.market_snapshot.config import AREA_BUCKETS

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import ALL_REGIONS, SEOUL_REGIONS


def _add_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    """aptSeq 컬럼에서 sggCd와 region_name, region_type을 파생한다.

    aptSeq 형식: "{sggCd}-{complexId}" (예: "11110-42")
    """
    df = df.copy()
    df["sggCd"] = df["aptSeq"].astype(str).str.split("-").str[0]
    df["region_name"] = df["sggCd"].map(ALL_REGIONS).fillna("기타")
    seoul_codes = set(SEOUL_REGIONS.keys())
    df["region_type"] = df["sggCd"].apply(
        lambda c: "서울" if c in seoul_codes else "경기"
    )
    return df


def _add_area_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """area 컬럼으로 면적 구간(area_bucket)을 생성한다."""
    df = df.copy()
    conditions = [
        (df["area"] < 60),
        (df["area"] >= 60) & (df["area"] < 85),
        (df["area"] >= 85) & (df["area"] < 102),
        (df["area"] >= 102),
    ]
    choices = [label for _, _, label in AREA_BUCKETS]
    df["area_bucket"] = np.select(conditions, choices, default="기타")
    return df


def _add_month_column(df: pd.DataFrame) -> pd.DataFrame:
    """date 컬럼에서 월 시작일(month)을 생성한다."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df
