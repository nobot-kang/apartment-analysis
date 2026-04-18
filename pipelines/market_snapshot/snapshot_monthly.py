"""A-1: 월별 거래량·중위 ㎡당 가격·분산 추이 집계."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def build_snapshot_monthly_trade(trade_df: pd.DataFrame) -> pd.DataFrame:
    """A-1용 매매 월별 집계 테이블을 생성한다.

    집계 단위: region_type × sggCd × month
    추가로 전체(ALL) 집계 행도 포함한다.

    Returns:
        month, sggCd, region_name, region_type,
        trade_count,
        price_median_m2, price_mean_m2, price_std_m2, price_p25_m2, price_p75_m2,
        price_median_total,
        rolling_3m_median_m2, rolling_6m_median_m2, rolling_12m_median_m2
    """
    logger.info("A-1 매매 월별 집계 생성 중...")
    df = trade_df.dropna(subset=["date", "price_per_m2"]).copy()

    def _agg_group(g: pd.DataFrame, sgg: str, name: str, rtype: str) -> dict:
        return {
            "sggCd": sgg,
            "region_name": name,
            "region_type": rtype,
            "trade_count": len(g),
            "price_median_m2": g["price_per_m2"].median(),
            "price_mean_m2": g["price_per_m2"].mean(),
            "price_std_m2": g["price_per_m2"].std(),
            "price_p25_m2": g["price_per_m2"].quantile(0.25),
            "price_p75_m2": g["price_per_m2"].quantile(0.75),
            "price_median_total": g["price"].median(),
        }

    # 지역별 × 월별 집계
    rows = []
    for (sggCd, region_name, region_type, month), g in df.groupby(
        ["sggCd", "region_name", "region_type", "month"]
    ):
        row = _agg_group(g, sggCd, region_name, region_type)
        row["month"] = month
        rows.append(row)

    # 전국 합계 행 (sggCd="ALL")
    for month, g in df.groupby("month"):
        row = _agg_group(g, "ALL", "전체", "전체")
        row["month"] = month
        rows.append(row)

    # 서울 합계 행
    seoul_df = df[df["region_type"] == "서울"]
    for month, g in seoul_df.groupby("month"):
        row = _agg_group(g, "SEOUL", "서울 전체", "서울")
        row["month"] = month
        rows.append(row)

    # 경기 합계 행
    gyeonggi_df = df[df["region_type"] == "경기"]
    for month, g in gyeonggi_df.groupby("month"):
        row = _agg_group(g, "GYEONGGI", "경기 전체", "경기")
        row["month"] = month
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["sggCd", "month"]).reset_index(drop=True)
    result["month"] = pd.to_datetime(result["month"])

    # Rolling 이동평균 (지역별 시계열 기준)
    for window, col in [
        (3, "rolling_3m_median_m2"),
        (6, "rolling_6m_median_m2"),
        (12, "rolling_12m_median_m2"),
    ]:
        result[col] = (
            result.groupby("sggCd", sort=False)["price_median_m2"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )

    return result


def build_snapshot_monthly_rent(rent_df: pd.DataFrame) -> pd.DataFrame:
    """A-1용 전월세 월별 집계 테이블을 생성한다.

    집계 단위: region_type × sggCd × rentType × month

    Returns:
        month, sggCd, region_name, region_type, rentType,
        rent_count,
        deposit_median_m2, deposit_mean_m2, deposit_std_m2,
        deposit_median_total,
        monthly_rent_median (월세만 유의미)
    """
    logger.info("A-1 전월세 월별 집계 생성 중...")
    df = rent_df.dropna(subset=["date"]).copy()

    rows = []

    for (sggCd, region_name, region_type, rent_type, month), g in df.groupby(
        ["sggCd", "region_name", "region_type", "rentType", "month"]
    ):
        rows.append({
            "month": month,
            "sggCd": sggCd,
            "region_name": region_name,
            "region_type": region_type,
            "rentType": rent_type,
            "rent_count": len(g),
            "deposit_median_m2": g["deposit_per_m2"].median() if "deposit_per_m2" in g.columns else np.nan,
            "deposit_mean_m2": g["deposit_per_m2"].mean() if "deposit_per_m2" in g.columns else np.nan,
            "deposit_std_m2": g["deposit_per_m2"].std() if "deposit_per_m2" in g.columns else np.nan,
            "deposit_median_total": g["deposit"].median(),
            "monthly_rent_median": g["monthly_rent"].median() if "monthly_rent" in g.columns else np.nan,
        })

    # 전체 집계 행 (rentType별)
    for (rent_type, month), g in df.groupby(["rentType", "month"]):
        rows.append({
            "month": month,
            "sggCd": "ALL",
            "region_name": "전체",
            "region_type": "전체",
            "rentType": rent_type,
            "rent_count": len(g),
            "deposit_median_m2": g["deposit_per_m2"].median() if "deposit_per_m2" in g.columns else np.nan,
            "deposit_mean_m2": g["deposit_per_m2"].mean() if "deposit_per_m2" in g.columns else np.nan,
            "deposit_std_m2": g["deposit_per_m2"].std() if "deposit_per_m2" in g.columns else np.nan,
            "deposit_median_total": g["deposit"].median(),
            "monthly_rent_median": g["monthly_rent"].median() if "monthly_rent" in g.columns else np.nan,
        })

    result = pd.DataFrame(rows).sort_values(["sggCd", "rentType", "month"]).reset_index(drop=True)
    result["month"] = pd.to_datetime(result["month"])
    return result
