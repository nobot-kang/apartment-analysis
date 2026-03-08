"""상관관계 분석 모듈.

시차(lag) 상관분석, 상관계수 행렬 생성, 회귀분석 등을 제공한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """지정 컬럼들의 상관계수 행렬을 계산한다.

    Args:
        df: 입력 DataFrame.
        columns: 상관분석 대상 컬럼 리스트. ``None`` 이면 모든 수치 컬럼.
        method: 상관계수 종류 (``pearson``, ``spearman``, ``kendall``).

    Returns:
        상관계수 행렬 DataFrame.
    """
    if columns is not None:
        subset = df[columns]
    else:
        subset = df.select_dtypes(include=[np.number])

    return subset.corr(method=method)


def lagged_correlation(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    max_lag: int = 12,
    method: str = "pearson",
) -> pd.DataFrame:
    """두 시계열 간 시차(lag) 상관계수를 계산한다.

    양의 lag는 ``col_x`` 가 ``col_y`` 보다 선행하는 경우,
    음의 lag는 ``col_y`` 가 선행하는 경우를 의미한다.

    Args:
        df: 입력 DataFrame (정렬된 시계열).
        col_x: 첫 번째 시계열 컬럼.
        col_y: 두 번째 시계열 컬럼.
        max_lag: 최대 시차 (양방향).
        method: 상관계수 종류.

    Returns:
        ``lag``, ``correlation`` 컬럼을 포함하는 DataFrame.
    """
    results: list[dict[str, float]] = []

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = df[col_x].iloc[:len(df) - lag] if lag > 0 else df[col_x]
            y = df[col_y].iloc[lag:]
        else:
            x = df[col_x].iloc[-lag:]
            y = df[col_y].iloc[:len(df) + lag]

        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # 둘 다 유효한 데이터가 있는 경우만 계산
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            corr = np.nan
        else:
            corr = x[mask].corr(y[mask], method=method)

        results.append({"lag": lag, "correlation": corr})

    return pd.DataFrame(results)


def simple_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> dict[str, float]:
    """단순 선형 회귀의 기울기, 절편, R-squared를 계산한다.

    Args:
        df: 입력 DataFrame.
        x_col: 독립변수 컬럼.
        y_col: 종속변수 컬럼.

    Returns:
        ``slope``, ``intercept``, ``r_squared`` 키를 포함하는 딕셔너리.
    """
    subset = df[[x_col, y_col]].dropna()

    if len(subset) < 2:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan}

    x = subset[x_col].values
    y = subset[y_col].values

    x_mean = x.mean()
    y_mean = y.mean()

    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
    ss_xx = ((x - x_mean) ** 2).sum()

    if ss_xx == 0:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan}

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()

    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}
