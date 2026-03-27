"""지역별 비교 분석 모듈.

자치구별 가격 비교, 상위/하위 순위, 지역 간 격차 분석 등을 제공한다.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SEOUL_REGIONS, GYEONGGI_REGIONS


def rank_regions_by_price(
    summary_df: pd.DataFrame,
    ym: str,
    price_col: str = "평균거래금액",
    region_col: str = "_region_name",
    ascending: bool = False,
) -> pd.DataFrame:
    """특정 월 기준 지역별 가격 순위를 반환한다.

    Args:
        summary_df: 월별 집계 DataFrame.
        ym: 대상 연월 ``YYYYMM``.
        price_col: 가격 컬럼명.
        region_col: 지역명 컬럼명.
        ascending: 오름차순 여부 (기본 내림차순 = 비싼 순).

    Returns:
        순위가 포함된 DataFrame.
    """
    filtered = summary_df[summary_df["ym"] == ym].copy()
    filtered = filtered.sort_values(price_col, ascending=ascending).reset_index(drop=True)
    filtered["순위"] = range(1, len(filtered) + 1)
    return filtered


def compare_regions(
    summary_df: pd.DataFrame,
    region_codes: list[str],
    price_col: str = "평균거래금액",
    region_col: str = "_lawd_cd",
) -> pd.DataFrame:
    """선택된 지역들의 시계열 데이터를 비교용으로 반환한다.

    Args:
        summary_df: 월별 집계 DataFrame.
        region_codes: 비교할 지역 코드 리스트.
        price_col: 가격 컬럼명.
        region_col: 지역 코드 컬럼명.

    Returns:
        ``ym`` × 지역코드별 가격 피벗 DataFrame.
    """
    filtered = summary_df[summary_df[region_col].isin(region_codes)].copy()
    pivoted = filtered.pivot_table(
        index="ym",
        columns="_region_name",
        values=price_col,
        aggfunc="first",
    ).reset_index()
    return pivoted


def price_gap_analysis(
    summary_df: pd.DataFrame,
    region_a: str,
    region_b: str,
    price_col: str = "평균거래금액",
    region_col: str = "_lawd_cd",
) -> pd.DataFrame:
    """두 지역 간 가격 격차 시계열을 계산한다.

    Args:
        summary_df: 월별 집계 DataFrame.
        region_a: 기준 지역 코드.
        region_b: 비교 지역 코드.
        price_col: 가격 컬럼명.
        region_col: 지역 코드 컬럼명.

    Returns:
        ``ym``, ``region_a``, ``region_b``, ``gap``, ``gap_pct`` 컬럼 DataFrame.
    """
    df_a = summary_df[summary_df[region_col] == region_a][["ym", price_col]].copy()
    df_b = summary_df[summary_df[region_col] == region_b][["ym", price_col]].copy()

    df_a = df_a.rename(columns={price_col: "price_a"})
    df_b = df_b.rename(columns={price_col: "price_b"})

    merged = df_a.merge(df_b, on="ym", how="inner")
    merged["gap"] = merged["price_a"] - merged["price_b"]
    merged["gap_pct"] = (merged["gap"] / merged["price_b"]) * 100

    return merged


def classify_region(code: str) -> str:
    """지역 코드를 '서울' 또는 '경기'로 분류한다.

    Args:
        code: 법정동 코드 5자리.

    Returns:
        ``"서울"`` 또는 ``"경기"`` 또는 ``"기타"``.
    """
    if code in SEOUL_REGIONS:
        return "서울"
    if code in GYEONGGI_REGIONS:
        return "경기"
    return "기타"


def aggregate_by_city(
    summary_df: pd.DataFrame,
    price_col: str = "평균거래금액",
    count_col: str = "거래건수",
) -> pd.DataFrame:
    """서울/경기 광역 단위로 집계한다.

    Args:
        summary_df: 월별 집계 DataFrame.
        price_col: 가격 컬럼명.
        count_col: 거래건수 컬럼명.

    Returns:
        서울/경기 월별 집계 DataFrame.
    """
    df = summary_df.copy()
    df["광역"] = df["_lawd_cd"].apply(classify_region)

    agg = (
        df.groupby(["ym", "광역"])
        .agg(
            총거래건수=(count_col, "sum"),
            가중평균거래금액=(price_col, "mean"),
        )
        .reset_index()
    )
    return agg
