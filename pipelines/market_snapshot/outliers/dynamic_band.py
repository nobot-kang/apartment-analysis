"""A-3 v2: 동적 band 폭 계산 (spread → sigma_eff → band_pct)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.market_snapshot.config import (
    SHRINK_K,
    BAND_Z,
    FLOOR_PCT_BASE,
    FLOOR_PCT_SPARSE_ADDON,
)


def _compute_dynamic_band(spreads_df: pd.DataFrame) -> pd.DataFrame:
    """G0 그룹별 동적 band 폭(band_pct)을 계산한다."""
    df = spreads_df.copy()

    # group MAD of path_g0 relative to ref_price_m2
    # We need path_g0_m2; reconstruct from spread_g0 + path_cohort_m2
    path_cohort = df["path_cohort_m2"].to_numpy(dtype=float)
    spread_g0 = df["spread_g0"].to_numpy(dtype=float)
    log_cohort = np.where(path_cohort > 0, np.log(path_cohort), np.nan)
    path_g0_m2 = np.where(
        np.isfinite(log_cohort) & np.isfinite(spread_g0),
        np.exp(log_cohort + spread_g0),
        np.nan,
    )
    df["_path_g0_m2"] = path_g0_m2

    def _group_mad(sub: pd.DataFrame) -> float:
        common = sub.dropna(subset=["_path_g0_m2", "ref_price_m2"])
        if len(common) < 2:
            return np.nan
        return float((common["_path_g0_m2"] - common["ref_price_m2"]).abs().median())

    group_mad_map: dict[tuple, float] = {}
    for keys, grp in df.groupby(["aptSeq", "area_repr"], observed=True):
        group_mad_map[keys] = _group_mad(grp)

    df["group_mad_m2"] = df.apply(
        lambda r: group_mad_map.get((r["aptSeq"], r["area_repr"]), np.nan), axis=1
    )

    # Cohort sigma should reflect local path volatility, not long-term level drift.
    # Using std(path level) over the whole history can explode when the market trends up/down.
    def _cohort_return_sigma_pct(sub: pd.DataFrame) -> float:
        unique_months = (
            sub[["month", "path_cohort_m2"]]
            .dropna(subset=["month", "path_cohort_m2"])
            .drop_duplicates(subset=["month"])
            .sort_values("month")
        )
        if len(unique_months) < 3:
            return np.nan

        log_path = np.log(unique_months["path_cohort_m2"].to_numpy(dtype=float))
        log_returns = np.diff(log_path)
        if log_returns.size < 2:
            return np.nan

        median_ret = float(np.median(log_returns))
        mad_ret = float(np.median(np.abs(log_returns - median_ret)))
        return mad_ret * 1.4826

    if "sggCd" in df.columns and "area_bucket" in df.columns:
        cohort_sigma_rows = []
        for (sgg, area_bucket), grp in df.groupby(["sggCd", "area_bucket"], observed=True):
            cohort_sigma_rows.append(
                {
                    "sggCd": sgg,
                    "area_bucket": area_bucket,
                    "cohort_return_sigma_pct": _cohort_return_sigma_pct(grp),
                }
            )
        cohort_sigma_pct = pd.DataFrame(cohort_sigma_rows)
        df = df.merge(cohort_sigma_pct, on=["sggCd", "area_bucket"], how="left")
    else:
        df["cohort_return_sigma_pct"] = np.nan

    ref_arr = df["ref_price_m2"].to_numpy(dtype=float)
    cohort_sigma_pct_arr = df["cohort_return_sigma_pct"].to_numpy(dtype=float)
    cohort_sigma_m2 = np.where(
        np.isfinite(cohort_sigma_pct_arr),
        cohort_sigma_pct_arr * ref_arr,
        np.nan,
    )
    df["cohort_sigma_m2"] = cohort_sigma_m2

    group_mad = df["group_mad_m2"].to_numpy(dtype=float)
    sigma_eff = np.where(
        np.isfinite(group_mad) & np.isfinite(cohort_sigma_m2),
        np.maximum(group_mad, cohort_sigma_m2),
        np.where(np.isfinite(group_mad), group_mad, cohort_sigma_m2),
    )
    df["sigma_eff"] = sigma_eff

    n_12m = df["n_12m"].to_numpy(dtype=float)
    is_leader = df["structure_type"] == "leader_or_isolated"
    floor_pct = np.where(n_12m <= SHRINK_K, FLOOR_PCT_BASE + FLOOR_PCT_SPARSE_ADDON, FLOOR_PCT_BASE)
    floor_pct = np.where(is_leader, floor_pct + 0.05, floor_pct)

    ref_safe = np.where(ref_arr > 0, ref_arr, np.nan)
    band_z_contrib = np.where(
        np.isfinite(sigma_eff) & np.isfinite(ref_safe),
        BAND_Z * sigma_eff / ref_safe,
        np.nan,
    )
    band_pct = np.where(
        np.isfinite(band_z_contrib),
        np.maximum(band_z_contrib, floor_pct),
        floor_pct,
    )
    df["band_pct"] = band_pct
    df["band_abs_m2"] = np.where(np.isfinite(ref_safe), band_pct * ref_safe, np.nan)

    cols_to_drop = [c for c in ["_path_g0_m2"] if c in df.columns]
    return df.drop(columns=cols_to_drop)
