"""A-3 v2: G0/G1/G2 스프레드 경로 및 shrinkage 혼합 기준가 계산."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.market_snapshot.config import SHRINK_K, LEADER_SPREAD_MONTHS, LEADER_SPREAD_SIGN_RATIO
from pipelines.market_snapshot.outliers._smoothing import smooth_log_series


def _build_complex_spreads(
    df: pd.DataFrame, c1_df: pd.DataFrame, c2_df: pd.DataFrame
) -> pd.DataFrame:
    """단지별 G0/G1/G2 스프레드 경로와 수축(shrinkage) 혼합 기준가를 계산한다."""
    # G0: aptSeq × area_repr × month
    g0_raw = (
        df.groupby(["aptSeq", "area_repr", "month"], observed=True)["price_per_m2"]
        .median()
        .reset_index()
        .rename(columns={"price_per_m2": "_g0_med"})
    )
    g0_parts = []
    for keys, grp in g0_raw.groupby(["aptSeq", "area_repr"], observed=True):
        g0_parts.append(smooth_log_series(grp, "_g0_med", "path_g0_m2"))
    g0_df = pd.concat(g0_parts, ignore_index=True)[["aptSeq", "area_repr", "month", "path_g0_m2"]]

    # G1: aptSeq × area_bucket × month
    df_with_bucket = df[["aptSeq", "area_bucket", "month", "price_per_m2"]].copy()
    g1_raw = (
        df_with_bucket.groupby(["aptSeq", "area_bucket", "month"], observed=True)["price_per_m2"]
        .median()
        .reset_index()
        .rename(columns={"price_per_m2": "_g1_med"})
    )
    g1_parts = []
    for keys, grp in g1_raw.groupby(["aptSeq", "area_bucket"], observed=True):
        g1_parts.append(smooth_log_series(grp, "_g1_med", "path_g1_m2"))
    g1_df = pd.concat(g1_parts, ignore_index=True)[["aptSeq", "area_bucket", "month", "path_g1_m2"]]

    # G2: aptSeq × month
    g2_raw = (
        df.groupby(["aptSeq", "month"], observed=True)["price_per_m2"]
        .median()
        .reset_index()
        .rename(columns={"price_per_m2": "_g2_med"})
    )
    g2_parts = []
    for apt, grp in g2_raw.groupby("aptSeq", observed=True):
        g2_parts.append(smooth_log_series(grp, "_g2_med", "path_g2_m2"))
    g2_df = pd.concat(g2_parts, ignore_index=True)[["aptSeq", "month", "path_g2_m2"]]

    # Merge sggCd and area_bucket onto G0 frame (take first value per aptSeq)
    meta = df[["aptSeq", "area_repr", "sggCd", "area_bucket"]].drop_duplicates(["aptSeq", "area_repr"])
    frame = g0_df.merge(meta, on=["aptSeq", "area_repr"], how="left")

    frame = frame.merge(g1_df, on=["aptSeq", "area_bucket", "month"], how="left")
    frame = frame.merge(g2_df, on=["aptSeq", "month"], how="left")
    frame = frame.merge(c1_df, on=["sggCd", "area_bucket", "month"], how="left")
    frame = frame.merge(c2_df, on=["sggCd", "month"], how="left")

    # Cohort reference: prefer C1, fallback C2
    frame["path_cohort_m2"] = frame["path_c1_m2"].where(frame["path_c1_m2"].notna(), frame["path_c2_m2"])
    frame["cohort_level"] = np.where(frame["path_c1_m2"].notna(), "C1", "C2")

    # Spreads in log space
    log_cohort = np.where(frame["path_cohort_m2"] > 0, np.log(frame["path_cohort_m2"]), np.nan)
    log_g0 = np.where(frame["path_g0_m2"] > 0, np.log(frame["path_g0_m2"]), np.nan)
    log_g1 = np.where(frame["path_g1_m2"] > 0, np.log(frame["path_g1_m2"]), np.nan)
    log_g2 = np.where(frame["path_g2_m2"] > 0, np.log(frame["path_g2_m2"]), np.nan)

    frame["spread_g0"] = np.where(
        np.isfinite(log_g0) & np.isfinite(log_cohort), log_g0 - log_cohort, np.nan
    )
    frame["spread_g1"] = np.where(
        np.isfinite(log_g1) & np.isfinite(log_cohort), log_g1 - log_cohort, np.nan
    )
    frame["spread_g2"] = np.where(
        np.isfinite(log_g2) & np.isfinite(log_cohort), log_g2 - log_cohort, np.nan
    )

    # n_12m: trailing 12-month G0 trade count per (aptSeq, area_repr)
    trade_counts_monthly = (
        df.groupby(["aptSeq", "area_repr", "month"], observed=True)
        .size()
        .reset_index(name="month_count")
    )
    trade_counts_monthly["month_ord"] = (
        trade_counts_monthly["month"].dt.year * 12 + trade_counts_monthly["month"].dt.month
    )

    def _trailing_12m_count(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.sort_values("month_ord").copy()
        counts = sub["month_count"].to_numpy(dtype=int)
        ords = sub["month_ord"].to_numpy(dtype=int)
        n12 = np.zeros(len(sub), dtype=int)
        for i in range(len(sub)):
            mask = (ords[i] - ords) < 12
            mask[i] = False
            n12[i] = int(counts[mask].sum()) + counts[i]
        sub["n_12m"] = n12
        return sub

    trailing_parts = []
    for keys, grp in trade_counts_monthly.groupby(["aptSeq", "area_repr"], observed=True):
        trailing_parts.append(_trailing_12m_count(grp))
    trailing_df = pd.concat(trailing_parts, ignore_index=True)[["aptSeq", "area_repr", "month", "n_12m", "month_count"]]
    frame = frame.merge(trailing_df, on=["aptSeq", "area_repr", "month"], how="left")
    frame["n_12m"] = frame["n_12m"].fillna(1).astype(int)
    frame["month_count"] = frame["month_count"].fillna(1).astype(int)

    # Shrinkage weight
    w0 = frame["n_12m"] / (frame["n_12m"] + SHRINK_K)

    # Best spread: coalesce G0, G1, G2, 0
    best_spread = frame["spread_g0"].where(
        frame["spread_g0"].notna(),
        frame["spread_g1"].where(frame["spread_g1"].notna(), frame["spread_g2"].fillna(0.0)),
    )
    fallback_spread = frame["spread_g1"].where(
        frame["spread_g1"].notna(),
        frame["spread_g2"].fillna(0.0),
    )
    frame["spread_shrunk"] = w0 * best_spread + (1.0 - w0) * fallback_spread

    # Reference price
    path_cohort_c2_fallback = frame["path_cohort_m2"].where(
        frame["path_cohort_m2"].notna(), frame["path_c2_m2"]
    )
    log_cohort_safe = np.where(
        path_cohort_c2_fallback > 0, np.log(path_cohort_c2_fallback), np.nan
    )
    frame["ref_price_m2"] = np.exp(
        np.where(np.isfinite(log_cohort_safe), log_cohort_safe + frame["spread_shrunk"].fillna(0.0), np.nan)
    )

    # Leader / isolated detection
    frame["structure_type"] = "normal"
    for (apt, area), grp in frame.groupby(["aptSeq", "area_repr"], observed=True):
        grp_sorted = grp.sort_values("month")
        month_ord = grp_sorted["month"].dt.year * 12 + grp_sorted["month"].dt.month
        spreads_g0 = grp_sorted["spread_g0"].to_numpy(dtype=float)
        max_ord = month_ord.max() if len(month_ord) else 0
        window_mask = (max_ord - month_ord.to_numpy()) < LEADER_SPREAD_MONTHS
        window_spreads = spreads_g0[window_mask]
        valid = window_spreads[np.isfinite(window_spreads)]
        if len(valid) >= 3:
            pos_ratio = np.mean(valid > 0)
            neg_ratio = np.mean(valid < 0)
            dominant_ratio = max(pos_ratio, neg_ratio)
            if dominant_ratio >= LEADER_SPREAD_SIGN_RATIO and abs(float(np.mean(valid))) > 0.15:
                frame.loc[grp.index, "structure_type"] = "leader_or_isolated"

    return frame[[
        "aptSeq", "area_repr", "month", "path_cohort_m2", "cohort_level",
        "spread_g0", "spread_g1", "spread_g2", "spread_shrunk", "ref_price_m2",
        "n_12m", "structure_type", "path_g1_m2", "month_count",
    ]]
