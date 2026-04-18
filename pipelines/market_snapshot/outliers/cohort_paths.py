"""A-3 v2: C1/C2 코호트 가격 경로 계산."""

from __future__ import annotations

import pandas as pd

from pipelines.market_snapshot.outliers._smoothing import smooth_log_series


def _build_cohort_paths(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C1 (sggCd × area_bucket × month) 및 C2 (sggCd × month) 코호트 가격 경로를 계산한다."""
    # C1: sggCd × area_bucket × month
    c1_raw = (
        df.groupby(["sggCd", "area_bucket", "month"], observed=True)["price_per_m2"]
        .median()
        .reset_index()
        .rename(columns={"price_per_m2": "_med"})
    )
    c1_parts = []
    for keys, grp in c1_raw.groupby(["sggCd", "area_bucket"], observed=True):
        c1_parts.append(smooth_log_series(grp, "_med", "path_c1_m2"))
    c1_df = pd.concat(c1_parts, ignore_index=True)[["sggCd", "area_bucket", "month", "path_c1_m2"]]

    # C2: sggCd × month
    c2_raw = (
        df.groupby(["sggCd", "month"], observed=True)["price_per_m2"]
        .median()
        .reset_index()
        .rename(columns={"price_per_m2": "_med"})
    )
    c2_parts = []
    for sgg, grp in c2_raw.groupby("sggCd", observed=True):
        c2_parts.append(smooth_log_series(grp, "_med", "path_c2_m2"))
    c2_df = pd.concat(c2_parts, ignore_index=True)[["sggCd", "month", "path_c2_m2"]]

    return c1_df, c2_df
