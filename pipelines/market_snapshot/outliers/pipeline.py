"""A-3 v2: 이상치 탐지 최종 조합 및 시세 테이블 생성 엔트리."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from pipelines.market_snapshot.config import (
    SANITY_LOG_RATIO,
    TREND_ROW_MIN_TRADE_COUNT,
    OLD_COMPLEX_AGE,
    RENOVATION_ABS_BUFFER_MANWON,
    RENOVATION_REL_CAP,
    ABS_DEVIATION_MANWON,
)
from pipelines.market_snapshot.preprocess import _add_region_columns, _add_area_bucket
from pipelines.market_snapshot.outliers.cohort_paths import _build_cohort_paths
from pipelines.market_snapshot.outliers.complex_spreads import _build_complex_spreads
from pipelines.market_snapshot.outliers.dynamic_band import _compute_dynamic_band
from pipelines.market_snapshot.outliers.trend_band import (
    _compute_monthly_band_frame,
    _annotate_trend_confirmation,
)


def build_snapshot_outliers(trade_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A-3용 이상치 탐지 및 단지별 월별 시세 테이블을 생성한다 (v2: spread-based).

    탐지 기준 (v2):
        그룹 단위   : aptSeq × area_repr
        기준 시세   : 코호트(sggCd × area_bucket) 대비 shrinkage-blended spread 경로
        band        : dynamic band = max(BAND_Z × sigma_eff, floor_pct)
        판정 방식   :
            1. sanity error: |log(price) - log(cohort)| > ln(2) AND |log(price) - log(ref)| > ln(2)
            2. band candidate: |price - ref| / ref > band_pct
            3. support confirmation: back 2M + forward 6M 지지 없으면 unsupported_jump
            4. trend-month row-level robust band: 추세 전환으로 인정된 월 내 개별 spike
            5. renovation buffer: 고령 단지 고가 이상치 완화
        제외 대상   : 1층 거래
    """
    logger.info("A-3 v2 이상치 탐지 시작 (spread-based)...")

    df = trade_df.dropna(subset=["date", "price_per_m2", "area_repr", "aptSeq"]).copy()
    df["month"] = pd.to_datetime(df["month"])

    floor_numeric = pd.to_numeric(df["floor"], errors="coerce")
    n_before = len(df)
    df = df[floor_numeric != 1].reset_index(drop=True)
    logger.info(f"  1층 거래 제외: {n_before - len(df):,}건 → {len(df):,}건")

    if "sggCd" not in df.columns:
        df = _add_region_columns(df)
    if "area_bucket" not in df.columns:
        df = _add_area_bucket(df)

    logger.info("  코호트 가격 경로 계산 중...")
    c1_df, c2_df = _build_cohort_paths(df)

    logger.info("  단지별 스프레드 계산 중...")
    spreads_df = _build_complex_spreads(df, c1_df, c2_df)

    # Attach sggCd/area_bucket to spreads_df for dynamic band computation
    meta = df[["aptSeq", "area_repr", "sggCd", "area_bucket"]].drop_duplicates(["aptSeq", "area_repr"])
    spreads_df = spreads_df.merge(meta, on=["aptSeq", "area_repr"], how="left")

    logger.info("  동적 band 계산 중...")
    spreads_df = _compute_dynamic_band(spreads_df)

    # Merge spreads onto individual trade rows
    merge_spread_cols = [
        "aptSeq", "area_repr", "month",
        "path_cohort_m2", "cohort_level",
        "spread_g0", "spread_g1", "spread_g2", "spread_shrunk",
        "ref_price_m2", "n_12m", "structure_type",
        "band_pct", "band_abs_m2",
        "path_g1_m2", "month_count",
    ]
    available_spread_cols = [c for c in merge_spread_cols if c in spreads_df.columns]
    evaluated = df.merge(spreads_df[available_spread_cols], on=["aptSeq", "area_repr", "month"], how="left")

    ppm = evaluated["price_per_m2"].to_numpy(dtype=float)
    ref_m2 = evaluated["ref_price_m2"].to_numpy(dtype=float)
    cohort_m2 = evaluated["path_cohort_m2"].to_numpy(dtype=float)
    band_pct_arr = evaluated["band_pct"].to_numpy(dtype=float)

    log_ppm = np.where(ppm > 0, np.log(ppm), np.nan)
    log_ref = np.where(ref_m2 > 0, np.log(ref_m2), np.nan)
    log_cohort = np.where(cohort_m2 > 0, np.log(cohort_m2), np.nan)

    # Stage 1 — sanity error
    sanity_mask = (
        np.isfinite(log_ref)
        & np.isfinite(log_cohort)
        & (np.abs(log_ppm - log_cohort) > SANITY_LOG_RATIO)
        & (np.abs(log_ppm - log_ref) > SANITY_LOG_RATIO)
    )

    # Stage 2 — band candidate
    ref_valid = np.isfinite(ref_m2) & (ref_m2 > 0)
    dev_ratio = np.where(ref_valid, np.abs(ppm - ref_m2) / ref_m2, np.nan)
    candidate_mask = (
        ref_valid
        & ~sanity_mask
        & np.isfinite(band_pct_arr)
        & (dev_ratio > band_pct_arr)
    )

    # Stage 3 — support confirmation for candidates
    # Build monthly median G0 prices per (aptSeq, area_repr, month) for support checking
    monthly_g0 = (
        evaluated.groupby(["aptSeq", "area_repr", "month"], observed=True)["price_per_m2"]
        .median()
        .reset_index()
        .rename(columns={"price_per_m2": "_monthly_med"})
    )
    monthly_g0["_month_ord"] = monthly_g0["month"].dt.year * 12 + monthly_g0["month"].dt.month

    is_supported = np.zeros(len(evaluated), dtype=bool)

    evaluated_with_monthly = evaluated.copy()
    evaluated_with_monthly["_row_idx"] = np.arange(len(evaluated))
    evaluated_with_monthly["_month_ord"] = evaluated_with_monthly["month"].dt.year * 12 + evaluated_with_monthly["month"].dt.month

    cand_indices = np.where(candidate_mask)[0]
    if len(cand_indices) > 0:
        group_index = monthly_g0.groupby(["aptSeq", "area_repr"], observed=True).indices
        monthly_g0_arr_map: dict[tuple, tuple] = {}
        for keys, grp_idx in group_index.items():
            sub = monthly_g0.iloc[grp_idx].sort_values("_month_ord")
            monthly_g0_arr_map[keys] = (sub["_month_ord"].to_numpy(dtype=int), sub["_monthly_med"].to_numpy(dtype=float))

        g1_monthly = (
            evaluated.groupby(["aptSeq", "area_bucket", "month"], observed=True)["price_per_m2"]
            .median()
            .reset_index()
            .rename(columns={"price_per_m2": "_g1_med"})
        ) if "area_bucket" in evaluated.columns else None

        for row_i in cand_indices:
            row = evaluated.iloc[row_i]
            apt = row["aptSeq"]
            area = row["area_repr"]
            month_ord = int(row["month"].year * 12 + row["month"].month)
            ref_val = float(ref_m2[row_i])
            bpct = float(band_pct_arr[row_i])
            direction = 1.0 if ppm[row_i] > ref_val else -1.0

            if not np.isfinite(ref_val) or not np.isfinite(bpct):
                continue

            key = (apt, area)
            if key not in monthly_g0_arr_map:
                continue

            ords, meds = monthly_g0_arr_map[key]
            support_count = 0

            # backward: 2 months
            back_mask = ((month_ord - ords) > 0) & ((month_ord - ords) <= 2)
            if np.any(back_mask):
                back_meds = meds[back_mask]
                back_support = direction * (back_meds - ref_val) >= bpct * ref_val * 0.5
                support_count += int(np.sum(back_support))

            # forward: 6 months
            fwd_mask = ((ords - month_ord) > 0) & ((ords - month_ord) <= 6)
            if np.any(fwd_mask):
                fwd_meds = meds[fwd_mask]
                fwd_support = direction * (fwd_meds - ref_val) >= bpct * ref_val * 0.5
                support_count += int(np.sum(fwd_support))

            if support_count >= 2:
                is_supported[row_i] = True
                continue

            # G1 confirmation
            if g1_monthly is not None and "area_bucket" in row.index:
                bucket = row["area_bucket"]
                g1_sub = g1_monthly[
                    (g1_monthly["aptSeq"] == apt) & (g1_monthly["area_bucket"] == bucket)
                ].sort_values("month")
                if len(g1_sub) >= 2:
                    g1_ords = (g1_sub["month"].dt.year * 12 + g1_sub["month"].dt.month).to_numpy(dtype=int)
                    g1_meds = g1_sub["_g1_med"].to_numpy(dtype=float)
                    adj_mask = (g1_ords != month_ord) & (np.abs(g1_ords - month_ord) <= 1)
                    if np.any(adj_mask):
                        g1_adj = g1_meds[adj_mask]
                        g1_ref_diff = direction * (g1_adj - ref_val)
                        if np.any(g1_ref_diff >= bpct * ref_val * 0.3):
                            is_supported[row_i] = True

    unsupported_mask = candidate_mask & ~is_supported

    # Stage 2b — absolute deviation outlier (±3억 원)
    ref_total_arr = ref_m2 * evaluated["area"].to_numpy(dtype=float)
    price_arr = evaluated["price"].to_numpy(dtype=float)
    abs_dev_manwon = np.abs(price_arr - ref_total_arr)
    abs_mask = (
        np.isfinite(ref_total_arr) & (ref_total_arr > 0)
        & np.isfinite(abs_dev_manwon)
        & (abs_dev_manwon >= ABS_DEVIATION_MANWON)
        & ~sanity_mask
        & ~unsupported_mask
    )

    # Build is_outlier from spread-based logic
    is_outlier_v2 = sanity_mask | unsupported_mask | abs_mask
    outlier_reason_v2 = np.where(
        sanity_mask, "sanity_error",
        np.where(unsupported_mask, "unsupported_jump",
        np.where(abs_mask, "abs_deviation", "")),
    )

    # Legacy trend-month band pass (to keep backward compat for trend_month_robust_band)
    logger.info("  추세 전환 월 row-level band 계산 중 (legacy)...")
    monthly_legacy = _compute_monthly_band_frame(df)
    monthly_legacy = _annotate_trend_confirmation(monthly_legacy)

    legacy_merge_cols = [
        "aptSeq", "area_repr", "month",
        "month_price_m2", "month_trade_count",
        "ref_month", "ref_price",
        "band_width_abs", "band_lower", "band_upper", "band_width_pct",
        "month_row_band_abs",
        "trend_confirmed", "trend_support_months", "trend_total_trades", "trend_ref_price",
    ]
    evaluated = evaluated.merge(monthly_legacy[legacy_merge_cols], on=["aptSeq", "area_repr", "month"], how="left")

    trend_row_outlier = (
        evaluated["trend_confirmed"].fillna(False)
        & evaluated["month_trade_count"].fillna(0).ge(TREND_ROW_MIN_TRADE_COUNT)
        & evaluated["month_price_m2"].notna()
        & (evaluated["price_per_m2"] - evaluated["month_price_m2"]).abs().gt(evaluated["month_row_band_abs"])
    )

    # Combine: spread-based OR trend-row outlier; trend-row takes precedence for reference_type
    is_outlier_final = is_outlier_v2 | trend_row_outlier.to_numpy()

    outlier_reason_final = outlier_reason_v2.copy()
    outlier_reason_final = np.where(
        trend_row_outlier.to_numpy() & ~is_outlier_v2,
        "trend_month_robust_band",
        outlier_reason_final,
    )

    reference_type_arr = np.where(
        trend_row_outlier.to_numpy(),
        "trend_month_robust_band",
        np.where(sanity_mask, "sanity_error", "moving_average_band"),
    )

    evaluated["is_outlier"] = is_outlier_final
    evaluated["outlier_reason"] = outlier_reason_final
    evaluated["reference_type"] = reference_type_arr
    evaluated["renovation_buffer_applied"] = False

    # Stage 4 — renovation buffer release
    raw_age = pd.to_numeric(
        evaluated.get("age", pd.Series(np.nan, index=evaluated.index)),
        errors="coerce",
    )
    age_arr = raw_age.fillna(0)
    construction_year = pd.to_numeric(
        evaluated.get("construction_year", pd.Series(0, index=evaluated.index)),
        errors="coerce",
    ).fillna(0)
    ref_total = ref_total_arr   # already computed above (만원 단위)
    dev_pct_arr = np.where(ref_m2 > 0, (ppm - ref_m2) / ref_m2 * 100.0, np.nan)

    reno_mask = (
        is_outlier_final
        & (ppm > ref_m2)
        & (age_arr.to_numpy() >= OLD_COMPLEX_AGE)
        & ((price_arr - ref_total) <= RENOVATION_ABS_BUFFER_MANWON)
        & (dev_pct_arr <= RENOVATION_REL_CAP * 100)
    )
    evaluated.loc[reno_mask, "is_outlier"] = False
    evaluated.loc[reno_mask, "renovation_buffer_applied"] = True

    # Stage 5 — exempt conditions (force-false): first snapshot month & age==0 신축
    from config.settings import START_YM  # function-local import for monkeypatch
    first_snapshot_period = pd.Period(f"{START_YM[:4]}-{START_YM[4:]}", freq="M")

    first_month_mask = (
        evaluated["month"].dt.to_period("M") == first_snapshot_period
    ).to_numpy()

    age_zero_exempt_mask = (
        raw_age.notna()
        & (raw_age == 0)
        & (construction_year > 0)
    ).to_numpy()

    exempt_condition_mask = age_zero_exempt_mask | first_month_mask
    live_is_outlier = evaluated["is_outlier"].to_numpy()
    exempt_flipped_mask = live_is_outlier & exempt_condition_mask

    evaluated.loc[exempt_flipped_mask, "is_outlier"] = False
    # renovation 공로 재정의: 면제 조건 만족 행은 renovation 공로에서 제외
    evaluated.loc[exempt_condition_mask, "renovation_buffer_applied"] = False

    # Compute final deviation columns
    effective_ref_price = np.where(
        trend_row_outlier.to_numpy(),
        evaluated["month_price_m2"].to_numpy(dtype=float),
        ref_m2,
    )
    evaluated["price_deviation_pct"] = np.where(
        effective_ref_price > 0,
        (ppm - effective_ref_price) / effective_ref_price * 100.0,
        np.nan,
    )
    evaluated["outlier_direction"] = pd.Series(pd.NA, index=evaluated.index, dtype="object")
    has_dev = evaluated["price_deviation_pct"].notna()
    evaluated.loc[has_dev & evaluated["price_deviation_pct"].gt(0), "outlier_direction"] = "고가이상치"
    evaluated.loc[has_dev & evaluated["price_deviation_pct"].le(0), "outlier_direction"] = "저가이상치"

    # Backward-compat: ref_price uses legacy band ref where applicable, else spread ref
    evaluated["ref_price"] = np.where(
        trend_row_outlier.to_numpy(),
        evaluated["month_price_m2"].to_numpy(dtype=float),
        ref_m2,
    )
    evaluated["band_width_pct"] = np.where(
        trend_row_outlier.to_numpy(),
        evaluated["month_row_band_abs"].fillna(np.nan) / np.where(evaluated["month_price_m2"] > 0, evaluated["month_price_m2"], np.nan) * 100.0,
        evaluated["band_pct"].fillna(np.nan) * 100.0,
    )

    # Diagnostic new columns
    evaluated["ref_price_shrunk"] = ref_m2
    evaluated["ref_price_total"] = ref_m2 * evaluated["area"].to_numpy(dtype=float)
    evaluated["deviation_total_krw"] = price_arr - evaluated["ref_price_total"].to_numpy(dtype=float)

    # trend_ref_price backward compat: use legacy value
    # (already merged from monthly_legacy)

    outliers_df = evaluated[evaluated["is_outlier"]].copy()

    keep_cols = [
        "month", "date", "aptSeq", "apt_name", "dong", "dong_repr",
        "area", "area_repr", "floor", "construction_year", "age",
        "price", "price_per_m2",
        "ref_price", "band_width_pct", "price_deviation_pct",
        "outlier_direction", "reference_type",
        "trend_confirmed", "trend_support_months", "trend_total_trades", "trend_ref_price",
        "path_cohort_m2", "spread_g0", "spread_g1", "spread_g2", "spread_shrunk",
        "ref_price_shrunk", "ref_price_total", "structure_type", "cohort_level",
        "deviation_total_krw", "renovation_buffer_applied", "outlier_reason",
    ]
    keep_cols = [c for c in keep_cols if c in outliers_df.columns]
    outliers_df = outliers_df[keep_cols].sort_values(["aptSeq", "area_repr", "month", "date"]).reset_index(drop=True)

    market_price_df = (
        evaluated[~evaluated["is_outlier"]]
        .groupby(["aptSeq", "area_repr", "month"], observed=True)
        .agg(
            market_price_m2=("price_per_m2", "median"),
            trade_count=("price_per_m2", "size"),
            renovation_buffer_count=("renovation_buffer_applied", "sum"),
        )
        .reset_index()
        .sort_values(["aptSeq", "area_repr", "month"])
        .reset_index(drop=True)
    )
    market_price_df["renovation_buffer_count"] = market_price_df["renovation_buffer_count"].astype(int)
    market_price_df["renovation_buffer_applied"] = market_price_df["renovation_buffer_count"] > 0
    market_price_df["month"] = pd.to_datetime(market_price_df["month"])

    n_total = len(evaluated)
    n_outlier = int(evaluated["is_outlier"].sum())
    n_sanity = int(np.sum(sanity_mask))
    n_unsupported = int(np.sum(unsupported_mask))
    n_abs = int(np.sum(abs_mask))
    n_trend_row = int(trend_row_outlier.sum())
    n_reno_raw = int(np.sum(reno_mask))
    n_reno_effective = int(evaluated["renovation_buffer_applied"].sum())
    n_first_month_exempt = int(first_month_mask.sum())
    n_age_zero_exempt = int(age_zero_exempt_mask.sum())
    n_exempt_condition = int(exempt_condition_mask.sum())
    n_exempt_flipped = int(exempt_flipped_mask.sum())
    logger.info(
        f"A-3 v2 완료: 이상치 {n_outlier:,}건 / {n_total:,}건 "
        f"({n_outlier / n_total * 100:.2f}%) | "
        f"sanity_error={n_sanity}, unsupported_jump={n_unsupported}, "
        f"abs_deviation={n_abs}, trend_month_robust_band={n_trend_row}, "
        f"renovation_buffer_raw={n_reno_raw}, renovation_buffer_effective={n_reno_effective}, "
        f"first_month_exempt={n_first_month_exempt}, age_zero_exempt={n_age_zero_exempt}, "
        f"exempt_condition_total={n_exempt_condition}, exempt_flipped={n_exempt_flipped}, "
        f"최종 시세 {len(market_price_df):,}행"
    )
    return outliers_df, market_price_df
