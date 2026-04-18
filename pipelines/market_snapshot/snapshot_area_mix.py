"""A-2: 면적 믹스 변화 & 구성효과 분해."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def build_snapshot_area_mix(trade_df: pd.DataFrame) -> pd.DataFrame:
    """A-2용 면적 구간별 거래 비중 및 구성효과 분해 테이블을 생성한다.

    구성효과 분해:
        - base_year 가중치(2020)로 고정가중 지수 vs 실제 가중 지수를 비교
        - composition_effect = weighted_mean - fixed_weight_mean

    Returns:
        month, sggCd, region_name, region_type, area_bucket,
        trade_count, share_pct,
        price_median_m2, price_mean_m2,
        fixed_weight_mean_m2 (기준년도 가중치 적용 전체 평균),
        actual_mean_m2 (실제 가중 평균),
        composition_effect_m2
    """
    logger.info("A-2 면적 믹스 집계 생성 중...")
    df = trade_df.dropna(subset=["date", "price_per_m2", "area_bucket"]).copy()

    # 면적 구간별 × 지역별 × 월별 집계
    grp = (
        df.groupby(["month", "sggCd", "region_name", "region_type", "area_bucket"])
        .agg(
            trade_count=("price_per_m2", "count"),
            price_median_m2=("price_per_m2", "median"),
            price_mean_m2=("price_per_m2", "mean"),
        )
        .reset_index()
    )

    # 각 month × sggCd 내 비중 계산
    total_per_group = grp.groupby(["month", "sggCd"])["trade_count"].transform("sum")
    grp["share_pct"] = (grp["trade_count"] / total_per_group * 100).round(2)

    # 구성효과 분해 (전체 기준, sggCd="ALL")
    base_year = 2020

    # 기준년도 면적 구간별 가중치
    base_mask = df["month"].dt.year == base_year
    base_weights = (
        df[base_mask].groupby("area_bucket")["price_per_m2"]
        .count()
        .rename("base_count")
    )
    if not base_weights.empty:
        base_weights = (base_weights / base_weights.sum()).rename("base_weight")
    else:
        base_weights = pd.Series(dtype=float)

    # 월별 전체 기준 구성효과 계산
    monthly_composition = []
    for month, mg in df.groupby("month"):
        bucket_mean = mg.groupby("area_bucket")["price_per_m2"].mean()
        actual_mean = mg["price_per_m2"].mean()

        fixed_weight_mean = np.nan
        if not base_weights.empty:
            shared = bucket_mean.index.intersection(base_weights.index)
            if len(shared) > 0:
                fixed_weight_mean = (
                    bucket_mean[shared] * base_weights[shared] / base_weights[shared].sum()
                ).sum()

        monthly_composition.append({
            "month": month,
            "actual_mean_m2": actual_mean,
            "fixed_weight_mean_m2": fixed_weight_mean,
            "composition_effect_m2": actual_mean - fixed_weight_mean if not np.isnan(fixed_weight_mean) else np.nan,
        })

    composition_df = pd.DataFrame(monthly_composition)
    grp = grp.merge(composition_df, on="month", how="left")
    grp["month"] = pd.to_datetime(grp["month"])

    return grp.sort_values(["month", "sggCd", "area_bucket"]).reset_index(drop=True)
