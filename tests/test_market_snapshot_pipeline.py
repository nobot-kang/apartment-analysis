from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.market_snapshot_pipeline import (
    _add_area_bucket,
    _add_region_columns,
    _build_cohort_paths,
    _build_complex_spreads,
    _compute_dynamic_band,
    build_snapshot_monthly_trade,
    build_snapshot_outliers,
)


def _build_trade_df(
    month_values: list[float | list[float]],
    *,
    apt_seq: str,
    area_repr: int = 84,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_idx, value in enumerate(month_values, start=1):
        prices = value if isinstance(value, list) else [value]
        for trade_idx, price_per_m2 in enumerate(prices, start=1):
            date = pd.Timestamp(2020, month_idx, min(5 + trade_idx, 28))
            rows.append(
                {
                    "date": date,
                    "month": date.to_period("M").to_timestamp(),
                    "aptSeq": apt_seq,
                    "area_repr": area_repr,
                    "price_per_m2": float(price_per_m2),
                    "price": int(float(price_per_m2) * 84),
                    "area": 84.0,
                    "floor": 10,
                    "construction_year": 2010,
                    "age": 10,
                    "dong": "테스트동",
                    "dong_repr": "테스트동(11110)",
                    "apt_name": "테스트아파트",
                }
            )
    return pd.DataFrame(rows)


def test_isolated_spike_stays_outlier() -> None:
    trade_df = _build_trade_df(
        [100, 101, 99, 102, 104, 160, 103, 102],
        apt_seq="test-iso",
    )

    outliers_df, market_price_df = build_snapshot_outliers(trade_df)

    assert len(outliers_df) == 1
    assert outliers_df.iloc[0]["month"] == pd.Timestamp("2020-06-01")
    assert outliers_df.iloc[0]["reference_type"] == "moving_average_band"
    assert market_price_df["month"].nunique() == 7


def test_persistent_shift_is_restored_as_trend() -> None:
    trade_df = _build_trade_df(
        [100, 101, 100, 102, 130, 131, 132],
        apt_seq="test-trend",
    )

    outliers_df, market_price_df = build_snapshot_outliers(trade_df)

    assert outliers_df.empty
    assert market_price_df["month"].nunique() == 7


def test_trend_month_still_filters_row_level_spike() -> None:
    trade_df = _build_trade_df(
        [100, 101, 100, 102, 130, 131, [132, 133, 170]],
        apt_seq="test-trend-row",
    )

    outliers_df, _ = build_snapshot_outliers(trade_df)

    assert len(outliers_df) == 1
    assert outliers_df.iloc[0]["reference_type"] == "trend_month_robust_band"
    assert outliers_df.iloc[0]["month"] == pd.Timestamp("2020-07-01")


def test_monthly_trade_rolling_follows_selected_region_series() -> None:
    trade_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-05",
                    "2020-02-05",
                    "2020-03-05",
                    "2020-01-05",
                    "2020-02-05",
                    "2020-03-05",
                ]
            ),
            "month": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                ]
            ),
            "price_per_m2": [100.0, 200.0, 300.0, 10.0, 20.0, 30.0],
            "price": [8400, 16800, 25200, 840, 1680, 2520],
            "sggCd": ["11110", "11110", "11110", "11200", "11200", "11200"],
            "region_name": ["종로구", "종로구", "종로구", "성동구", "성동구", "성동구"],
            "region_type": ["서울", "서울", "서울", "서울", "서울", "서울"],
        }
    )

    result = build_snapshot_monthly_trade(trade_df)

    jongno = result[result["sggCd"] == "11110"].sort_values("month")
    seongdong = result[result["sggCd"] == "11200"].sort_values("month")

    assert jongno["rolling_3m_median_m2"].round(4).tolist() == [100.0, 150.0, 200.0]
    assert seongdong["rolling_3m_median_m2"].round(4).tolist() == [10.0, 15.0, 20.0]
    assert jongno["rolling_12m_median_m2"].round(4).tolist() == [100.0, 150.0, 200.0]
    assert seongdong["rolling_12m_median_m2"].round(4).tolist() == [10.0, 15.0, 20.0]


# ---------------------------------------------------------------------------
# A-3 v2 tests
# ---------------------------------------------------------------------------

def _build_trade_df_multi(
    specs: list[dict],
) -> pd.DataFrame:
    """Build a trade DataFrame from a list of spec dicts.

    Each spec: {apt_seq, area_repr, month_values, area, age, construction_year}
    month_values: same format as _build_trade_df
    """
    rows: list[dict] = []
    for spec in specs:
        apt_seq: str = spec["apt_seq"]
        area_repr: int = spec.get("area_repr", 84)
        area: float = spec.get("area", 84.0)
        age: int = spec.get("age", 10)
        construction_year: int = spec.get("construction_year", 2010)
        month_values = spec["month_values"]
        for month_idx, value in enumerate(month_values, start=1):
            prices = value if isinstance(value, list) else [value]
            for trade_idx, price_per_m2 in enumerate(prices, start=1):
                date = pd.Timestamp(2020, month_idx, min(5 + trade_idx, 28))
                rows.append(
                    {
                        "date": date,
                        "month": date.to_period("M").to_timestamp(),
                        "aptSeq": apt_seq,
                        "area_repr": area_repr,
                        "price_per_m2": float(price_per_m2),
                        "price": int(float(price_per_m2) * area),
                        "area": area,
                        "floor": 10,
                        "construction_year": construction_year,
                        "age": age,
                        "dong": "테스트동",
                        "dong_repr": "테스트동(11110)",
                        "apt_name": "테스트아파트",
                    }
                )
    return pd.DataFrame(rows)


def test_sparse_group_uses_cohort_fallback() -> None:
    """Sparse apt with only 2 trades: the spike should be flagged via cohort reference."""
    # main-apt: stable ~100 for months 1-12 → establishes cohort price ~100
    main_prices = [100, 101, 99, 100, 102, 100, 101, 99, 100, 101, 100, 99]
    # sparse-apt: 1 normal trade at month 6, 1 spike at month 7 (price 250)
    sparse_prices = [None] * 5 + [100, 250] + [None] * 5

    main_rows = []
    for i, p in enumerate(main_prices, start=1):
        date = pd.Timestamp(2020, i, 10)
        main_rows.append({
            "date": date,
            "month": date.to_period("M").to_timestamp(),
            "aptSeq": "11110-main",
            "area_repr": 84,
            "price_per_m2": float(p),
            "price": int(p * 84),
            "area": 84.0,
            "floor": 10,
            "construction_year": 2010,
            "age": 10,
            "dong": "테스트동",
            "dong_repr": "테스트동(11110)",
            "apt_name": "메인아파트",
        })

    sparse_rows = []
    for i, p in enumerate(sparse_prices, start=1):
        if p is None:
            continue
        date = pd.Timestamp(2020, i, 15)
        sparse_rows.append({
            "date": date,
            "month": date.to_period("M").to_timestamp(),
            "aptSeq": "11110-sparse",
            "area_repr": 84,
            "price_per_m2": float(p),
            "price": int(p * 84),
            "area": 84.0,
            "floor": 10,
            "construction_year": 2010,
            "age": 10,
            "dong": "테스트동",
            "dong_repr": "테스트동(11110)",
            "apt_name": "희소아파트",
        })

    trade_df = pd.DataFrame(main_rows + sparse_rows)
    outliers_df, _ = build_snapshot_outliers(trade_df)

    sparse_outliers = outliers_df[outliers_df["aptSeq"] == "11110-sparse"]
    assert len(sparse_outliers) >= 1, "Spike from sparse-apt should be flagged"
    assert pd.Timestamp("2020-07-01") in sparse_outliers["month"].values


def test_leader_complex_not_flagged_as_outlier() -> None:
    """Leader complex consistently 2x cohort price — month-19 trade should NOT be flagged."""
    # normal-apt: ~100 for 19 months
    normal_prices = [100.0] * 19
    # leader-apt: ~200 for months 1-18, then 200 at month 19
    leader_prices = [200.0] * 19

    rows = []
    for apt, prices in [("11110-normal", normal_prices), ("11110-leader", leader_prices)]:
        for i, p in enumerate(prices, start=1):
            date = pd.Timestamp(2020, min(i, 12), min(i if i <= 12 else i - 12, 28))
            # Spread months across 2020-2021
            year = 2020 + (i - 1) // 12
            month = ((i - 1) % 12) + 1
            date = pd.Timestamp(year, month, 10)
            rows.append({
                "date": date,
                "month": date.to_period("M").to_timestamp(),
                "aptSeq": apt,
                "area_repr": 84,
                "price_per_m2": float(p),
                "price": int(p * 84),
                "area": 84.0,
                "floor": 10,
                "construction_year": 2010,
                "age": 10,
                "dong": "테스트동",
                "dong_repr": "테스트동(11110)",
                "apt_name": "테스트아파트",
            })

    trade_df = pd.DataFrame(rows)
    outliers_df, _ = build_snapshot_outliers(trade_df)

    leader_outliers = outliers_df[outliers_df["aptSeq"] == "11110-leader"]
    # Month 19 corresponds to year=2021, month=7
    month_19 = pd.Timestamp(2021, 7, 1)
    month_19_outliers = leader_outliers[leader_outliers["month"] == month_19]
    assert len(month_19_outliers) == 0, (
        f"Leader complex month-19 trade should NOT be flagged, got: {month_19_outliers}"
    )


def test_renovation_buffer_releases_high_outlier() -> None:
    """Old complex, +8% above ref within 50M KRW abs threshold → buffer releases outlier."""
    # age=25, area=84, price_per_m2 ~119 (= ref ~110 * 1.08), price = 119 * 84 ≈ 9,996,000 KRW
    # Use price_per_m2=110 for stable months, then 119 for month 9 (8% above ~110)
    # The absolute diff: (119 - 110) * 84 = 756万 KRW = 7,560,000 KRW < 50,000,000 KRW
    stable = [110.0] * 8
    spike_month = [119.0]  # ~8% above stable level

    trade_df = _build_trade_df_multi([
        {
            "apt_seq": "11110-reno",
            "area_repr": 84,
            "area": 84.0,
            "age": 25,
            "construction_year": 1995,
            "month_values": stable + spike_month,
        }
    ])

    outliers_df, _ = build_snapshot_outliers(trade_df)
    reno_outliers = outliers_df[outliers_df["aptSeq"] == "11110-reno"]
    month_9 = pd.Timestamp(2020, 9, 1)
    month_9_outliers = reno_outliers[reno_outliers["month"] == month_9]
    assert len(month_9_outliers) == 0, (
        "Renovation buffer should release month-9 high outlier for old complex within abs threshold"
    )


def test_renovation_buffer_does_not_release_downward() -> None:
    """Downward spike in old complex should still be flagged (buffer only applies upward)."""
    # age=25, price drops to 40 (relative to stable ~110)
    stable = [110.0] * 8
    low_month = [40.0]  # far below — should be outlier

    trade_df = _build_trade_df_multi([
        {
            "apt_seq": "11110-reno-down",
            "area_repr": 84,
            "area": 84.0,
            "age": 25,
            "construction_year": 1995,
            "month_values": stable + low_month,
        }
    ])

    outliers_df, _ = build_snapshot_outliers(trade_df)
    reno_outliers = outliers_df[outliers_df["aptSeq"] == "11110-reno-down"]
    month_9 = pd.Timestamp(2020, 9, 1)
    month_9_outliers = reno_outliers[reno_outliers["month"] == month_9]
    assert len(month_9_outliers) >= 1, (
        "Downward spike should still be flagged even for old complex"
    )


def test_dynamic_band_does_not_explode_from_cohort_level_trend() -> None:
    """Long-run level trend in cohort path should not inflate band volatility."""
    rows: list[dict[str, object]] = []
    for month_idx in range(1, 13):
        base_price = 100.0 + (month_idx - 1) * 10.0
        for apt_seq, multiplier in [("11110-base", 1.00), ("11110-target", 1.05)]:
            date = pd.Timestamp(2020, month_idx, 10)
            ppm = base_price * multiplier
            rows.append(
                {
                    "date": date,
                    "month": date.to_period("M").to_timestamp(),
                    "aptSeq": apt_seq,
                    "area_repr": 84,
                    "price_per_m2": ppm,
                    "price": int(ppm * 84),
                    "area": 84.0,
                    "floor": 10,
                    "construction_year": 2010,
                    "age": 10,
                    "dong": "테스트동",
                    "dong_repr": "테스트동(11110)",
                    "apt_name": "테스트아파트",
                }
            )

    trade_df = pd.DataFrame(rows)
    trade_df = _add_region_columns(trade_df)
    trade_df = _add_area_bucket(trade_df)

    c1_df, c2_df = _build_cohort_paths(trade_df)
    spreads_df = _build_complex_spreads(trade_df, c1_df, c2_df)
    meta = trade_df[["aptSeq", "area_repr", "sggCd", "area_bucket"]].drop_duplicates(["aptSeq", "area_repr"])
    spreads_df = spreads_df.merge(meta, on=["aptSeq", "area_repr"], how="left")
    spreads_df = _compute_dynamic_band(spreads_df)

    target_row = spreads_df[
        (spreads_df["aptSeq"] == "11110-target")
        & (spreads_df["area_repr"] == 84)
        & (spreads_df["month"] == pd.Timestamp("2020-12-01"))
    ].iloc[0]

    assert target_row["band_pct"] < 0.40, (
        "Band should reflect local cohort volatility, not full-history level drift"
    )
    assert target_row["cohort_sigma_m2"] < target_row["ref_price_m2"] * 0.10
