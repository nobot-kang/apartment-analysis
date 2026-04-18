"""A-3 legacy: Bollinger band + 추세 전환 확인 (trend_month_robust_band)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipelines.market_snapshot.config import (
    BOLLINGER_WINDOW_MONTHS,
    BOLLINGER_MIN_HISTORY_MONTHS,
    BOLLINGER_STD_MULTIPLIER,
    LOOKBACK_MONTHS,
    OUTLIER_THRESHOLD,
    TREND_LOOKAHEAD_MONTHS,
    TREND_MIN_SUPPORT_MONTHS,
    TREND_MIN_TOTAL_TRADES,
    TREND_SUPPORT_BAND_RATIO,
    TREND_ALIGNMENT_TOLERANCE,
    TREND_ROW_STD_MULTIPLIER,
    TREND_ROW_MIN_BAND_PCT,
)


def _compute_monthly_band_frame(df: pd.DataFrame) -> pd.DataFrame:
    """단지×면적×월 대표가격과 moving-average band 를 계산한다 (legacy, trend 확인용)."""
    group_cols = ["aptSeq", "area_repr", "month"]
    monthly = (
        df.groupby(group_cols, observed=True, sort=True)["price_per_m2"]
        .agg(
            month_price_m2="median",
            month_trade_count="size",
            month_price_std_m2=lambda s: float(s.std(ddof=0)) if len(s) > 1 else 0.0,
            month_price_mad_m2=lambda s: float((s - s.median()).abs().median()),
        )
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    group_keys = ["aptSeq", "area_repr"]
    monthly["_month_ord"] = monthly["month"].dt.year * 12 + monthly["month"].dt.month
    monthly["lag_price"] = monthly.groupby(group_keys, sort=False)["month_price_m2"].shift(1)
    monthly["ref_price"] = monthly.groupby(group_keys, sort=False)["lag_price"].transform(
        lambda s: s.rolling(BOLLINGER_WINDOW_MONTHS, min_periods=BOLLINGER_MIN_HISTORY_MONTHS).mean()
    )
    monthly["rolling_std_m2"] = monthly.groupby(group_keys, sort=False)["lag_price"].transform(
        lambda s: s.rolling(BOLLINGER_WINDOW_MONTHS, min_periods=BOLLINGER_MIN_HISTORY_MONTHS).std(ddof=0)
    )
    monthly["ref_month"] = monthly.groupby(group_keys, sort=False)["month"].shift(1)
    monthly["ref_month_ord"] = monthly.groupby(group_keys, sort=False)["_month_ord"].shift(1)
    monthly["ref_gap_months"] = monthly["_month_ord"] - monthly["ref_month_ord"]

    stale_ref = monthly["ref_gap_months"] > LOOKBACK_MONTHS
    monthly.loc[stale_ref, "ref_price"] = np.nan
    monthly.loc[stale_ref, "rolling_std_m2"] = np.nan
    monthly.loc[stale_ref, "ref_month"] = pd.NaT

    monthly["month_price_std_m2"] = monthly["month_price_std_m2"].fillna(0.0)
    monthly["month_price_mad_m2"] = monthly["month_price_mad_m2"].fillna(0.0)
    monthly["month_robust_sigma_m2"] = monthly["month_price_mad_m2"] * 1.4826
    monthly["month_row_band_abs"] = np.maximum(
        monthly["month_robust_sigma_m2"] * TREND_ROW_STD_MULTIPLIER,
        monthly["month_price_m2"] * TREND_ROW_MIN_BAND_PCT,
    )

    band_candidate = np.maximum(
        monthly["rolling_std_m2"].fillna(0.0) * BOLLINGER_STD_MULTIPLIER,
        monthly["ref_price"] * OUTLIER_THRESHOLD,
    )
    monthly["band_width_abs"] = np.where(monthly["ref_price"].notna(), band_candidate, np.nan)
    monthly["band_lower"] = monthly["ref_price"] - monthly["band_width_abs"]
    monthly["band_upper"] = monthly["ref_price"] + monthly["band_width_abs"]
    monthly["band_width_pct"] = np.where(
        monthly["ref_price"] > 0,
        monthly["band_width_abs"] / monthly["ref_price"] * 100,
        np.nan,
    )

    monthly["candidate_direction"] = 0
    monthly.loc[
        monthly["ref_price"].notna() & monthly["month_price_m2"].gt(monthly["band_upper"]),
        "candidate_direction",
    ] = 1
    monthly.loc[
        monthly["ref_price"].notna() & monthly["month_price_m2"].lt(monthly["band_lower"]),
        "candidate_direction",
    ] = -1
    return monthly


def _annotate_trend_confirmation(monthly: pd.DataFrame) -> pd.DataFrame:
    """후행 거래가 이어지는 breakout 월을 추세 전환으로 태깅한다."""
    monthly = monthly.copy()
    trend_confirmed = np.zeros(len(monthly), dtype=bool)
    trend_support_months = np.zeros(len(monthly), dtype=np.int16)
    trend_total_trades = np.zeros(len(monthly), dtype=np.int32)
    trend_ref_price = np.full(len(monthly), np.nan, dtype=float)

    group_indices = monthly.groupby(["aptSeq", "area_repr"], sort=False).indices

    for idx in tqdm(group_indices.values(), desc="Confirming trend shifts"):
        group_idx = np.asarray(idx)
        if group_idx.size <= BOLLINGER_MIN_HISTORY_MONTHS:
            continue

        months_ord = monthly.loc[group_idx, "_month_ord"].to_numpy(dtype=int)
        prices = monthly.loc[group_idx, "month_price_m2"].to_numpy(dtype=float)
        refs = monthly.loc[group_idx, "ref_price"].to_numpy(dtype=float)
        bands = monthly.loc[group_idx, "band_width_abs"].to_numpy(dtype=float)
        directions = monthly.loc[group_idx, "candidate_direction"].to_numpy(dtype=int)
        trade_counts = monthly.loc[group_idx, "month_trade_count"].to_numpy(dtype=int)

        for pos in range(group_idx.size):
            direction = directions[pos]
            ref_price = refs[pos]
            band_width = bands[pos]

            if direction == 0 or not np.isfinite(ref_price) or not np.isfinite(band_width):
                continue

            forward_gap = months_ord[pos + 1:] - months_ord[pos]
            if forward_gap.size == 0:
                continue

            future_positions = np.flatnonzero(forward_gap <= TREND_LOOKAHEAD_MONTHS) + pos + 1
            if future_positions.size == 0:
                continue

            support_positions = future_positions[
                direction * (prices[future_positions] - ref_price) >= band_width * TREND_SUPPORT_BAND_RATIO
            ]
            support_months = int(support_positions.size)
            total_trades = int(trade_counts[pos] + trade_counts[support_positions].sum())

            if support_months < TREND_MIN_SUPPORT_MONTHS or total_trades < TREND_MIN_TOTAL_TRADES:
                continue

            level_positions = np.concatenate(([pos], support_positions))
            new_level = float(np.average(prices[level_positions], weights=trade_counts[level_positions]))
            if not np.isfinite(new_level) or new_level <= 0:
                continue

            if abs(prices[pos] - new_level) / new_level > TREND_ALIGNMENT_TOLERANCE:
                continue

            sequence_idx = group_idx[level_positions]
            trend_confirmed[sequence_idx] = True
            for row_idx in sequence_idx:
                if support_months >= trend_support_months[row_idx]:
                    trend_support_months[row_idx] = support_months
                    trend_total_trades[row_idx] = max(trend_total_trades[row_idx], total_trades)
                    trend_ref_price[row_idx] = new_level

    monthly["trend_confirmed"] = trend_confirmed
    monthly["trend_support_months"] = trend_support_months
    monthly["trend_total_trades"] = trend_total_trades
    monthly["trend_ref_price"] = trend_ref_price
    return monthly
