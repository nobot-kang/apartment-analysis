"""트렌드 및 이동평균 분석 모듈.

시계열 데이터에 대한 이동평균, YoY(전년동월비), MoM(전월비) 계산 함수를 제공한다.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


def moving_average(
    df: pd.DataFrame,
    value_col: str,
    window: int = 3,
    min_periods: int = 1,
) -> pd.Series:
    """단순 이동평균을 계산한다.

    Args:
        df: 시계열 DataFrame (정렬되어 있어야 함).
        value_col: 이동평균을 계산할 컬럼명.
        window: 이동평균 기간 (기본 3개월).
        min_periods: 최소 관측치 수.

    Returns:
        이동평균 Series.
    """
    return df[value_col].rolling(window=window, min_periods=min_periods).mean()


def yoy_change(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "ym",
    method: Literal["pct", "diff"] = "pct",
) -> pd.Series:
    """전년동월비(YoY) 변화율 또는 차이를 계산한다.

    ``date_col`` 이 ``YYYYMM`` 형식이거나 datetime 이면 12개월 전과 비교한다.

    Args:
        df: 시계열 DataFrame (월별 정렬).
        value_col: 비교 대상 컬럼명.
        date_col: 연월 컬럼명.
        method: ``"pct"`` 이면 백분율 변화, ``"diff"`` 이면 절대 차이.

    Returns:
        YoY 변화 Series.
    """
    if method == "pct":
        return df[value_col].pct_change(periods=12) * 100
    return df[value_col].diff(periods=12)


def mom_change(
    df: pd.DataFrame,
    value_col: str,
    method: Literal["pct", "diff"] = "pct",
) -> pd.Series:
    """전월비(MoM) 변화율 또는 차이를 계산한다.

    Args:
        df: 시계열 DataFrame (월별 정렬).
        value_col: 비교 대상 컬럼명.
        method: ``"pct"`` 이면 백분율 변화, ``"diff"`` 이면 절대 차이.

    Returns:
        MoM 변화 Series.
    """
    if method == "pct":
        return df[value_col].pct_change(periods=1) * 100
    return df[value_col].diff(periods=1)


def add_trend_columns(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "ym",
    ma_windows: list[int] | None = None,
) -> pd.DataFrame:
    """이동평균, YoY, MoM 컬럼을 한꺼번에 추가한다.

    Args:
        df: 시계열 DataFrame.
        value_col: 대상 값 컬럼명.
        date_col: 연월 컬럼명.
        ma_windows: 이동평균 기간 리스트 (기본 [3, 6, 12]).

    Returns:
        트렌드 컬럼이 추가된 DataFrame (원본 변경 안 함).
    """
    if ma_windows is None:
        ma_windows = [3, 6, 12]

    result = df.copy()

    for w in ma_windows:
        result[f"{value_col}_MA{w}"] = moving_average(result, value_col, window=w)

    result[f"{value_col}_YoY"] = yoy_change(result, value_col, date_col)
    result[f"{value_col}_MoM"] = mom_change(result, value_col)

    return result
