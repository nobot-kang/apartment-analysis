from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from loguru import logger as _loguru_logger

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


# ===========================================================================
# Outlier classification v2 update — tests 1-12
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def loguru_records():
    """Capture loguru INFO-level messages."""
    records: list[str] = []
    handle = _loguru_logger.add(records.append, level="INFO", format="{message}")
    yield records
    _loguru_logger.remove(handle)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _month_seq(start: str, n: int) -> list[pd.Timestamp]:
    """Return n monthly timestamps starting from 'YYYY-MM'."""
    base = pd.Period(start, freq="M")
    return [(base + i).to_timestamp() for i in range(n)]


def _flat_rows(
    apt_seq: str,
    ppm: float,
    month_ts_list: list[pd.Timestamp],
    *,
    area: float = 84.0,
    age: object = 10,
    cy: int = 2010,
    n: int = 2,
) -> list[dict]:
    """n trades per month at ppm. age may be np.nan."""
    rows = []
    prefix = apt_seq.split("-")[0]
    for m in month_ts_list:
        for i in range(n):
            rows.append({
                "date": m + pd.Timedelta(days=5 + i),
                "month": m,
                "aptSeq": apt_seq,
                "area_repr": int(area),
                "price_per_m2": float(ppm),
                "price": int(ppm * area),
                "area": float(area),
                "floor": 10,
                "construction_year": cy,
                "age": age,
                "dong": "테스트동",
                "dong_repr": f"테스트동({prefix})",
                "apt_name": "테스트아파트",
            })
    return rows


def _single_row(
    apt_seq: str,
    *,
    ppm: float | None = None,
    price_manwon: int | None = None,
    month_ts: pd.Timestamp,
    area: float = 84.0,
    age: object = 10,
    cy: int = 2010,
    day: int = 20,
) -> dict:
    """One trade row; specify either ppm or price_manwon."""
    prefix = apt_seq.split("-")[0]
    if price_manwon is not None:
        p = int(price_manwon)
        ppm_val = p / area
    else:
        assert ppm is not None
        ppm_val = float(ppm)
        p = int(ppm_val * area)
    return {
        "date": month_ts + pd.Timedelta(days=day - 1),
        "month": month_ts,
        "aptSeq": apt_seq,
        "area_repr": int(area),
        "price_per_m2": ppm_val,
        "price": p,
        "area": float(area),
        "floor": 10,
        "construction_year": cy,
        "age": age,
        "dong": "테스트동",
        "dong_repr": f"테스트동({prefix})",
        "apt_name": "테스트아파트",
    }


def _build_trend_reno_df(
    new_level_ppm: float,
    delta_manwon: int,
    *,
    area: float = 100.0,
    age: int = 25,
    cy: int = 1990,
    start_month: str = "2020-01",
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Fixture for renovation buffer tests via trend_month_robust_band path.

    Month layout (offsets from start_month):
      0-6  : baseline at baseline_ppm (7 months, 2 trades each, both apts)
      7    : breakout at new_level (2 trades each)
      8    : target — anchor 2 normal + reno-test 2 normal + 1 row outlier
      9    : extra support at new_level (2 trades each)

    Both "11110-anchor" and "11110-reno-test" share the same cohort so that
    cohort path shifts to new_level, keeping spread-based ref_m2 ≈ new_level.
    Returns (trade_df, target_month_ts).
    """
    baseline_ppm = new_level_ppm / 1.30
    months = _month_seq(start_month, 10)
    prefix = "11110"
    rows: list[dict] = []

    def _row(apt, p, m_ts, day=10):
        return {
            "date": m_ts + pd.Timedelta(days=day - 1),
            "month": m_ts,
            "aptSeq": apt,
            "area_repr": int(area),
            "price_per_m2": float(p),
            "price": int(p * area),
            "area": float(area),
            "floor": 10,
            "construction_year": cy,
            "age": age,
            "dong": "테스트동",
            "dong_repr": f"테스트동({prefix})",
            "apt_name": "테스트아파트",
        }

    for apt in ("11110-anchor", "11110-reno-test"):
        for m in months[:7]:
            rows += [_row(apt, baseline_ppm, m, 5), _row(apt, baseline_ppm, m, 10)]
        # breakout
        rows += [_row(apt, new_level_ppm, months[7], 5), _row(apt, new_level_ppm, months[7], 10)]
        # target month — 2 normal trades
        rows += [_row(apt, new_level_ppm, months[8], 5), _row(apt, new_level_ppm, months[8], 10)]
        # extra support
        rows += [_row(apt, new_level_ppm, months[9], 5), _row(apt, new_level_ppm, months[9], 10)]

    # Target row for reno-test (price_manwon injection)
    target_price = int(new_level_ppm * area) + delta_manwon
    rows.append({
        "date": months[8] + pd.Timedelta(days=20),
        "month": months[8],
        "aptSeq": "11110-reno-test",
        "area_repr": int(area),
        "price_per_m2": target_price / area,
        "price": target_price,
        "area": float(area),
        "floor": 10,
        "construction_year": cy,
        "age": age,
        "dong": "테스트동",
        "dong_repr": f"테스트동({prefix})",
        "apt_name": "테스트아파트",
    })
    return pd.DataFrame(rows), months[8]


# ---------------------------------------------------------------------------
# Test 1: first snapshot month is exempt
# ---------------------------------------------------------------------------

def test_exempt_first_month(monkeypatch) -> None:
    """First snapshot month outlier is force-false; row still counts in market_price_df."""
    monkeypatch.setattr("config.settings.START_YM", "202006")
    months = _month_seq("2020-01", 12)
    rows = (
        _flat_rows("11110-anc", 100.0, months)
        + _flat_rows("11110-b", 100.0, months[:5])
        + _flat_rows("11110-b", 100.0, months[6:])
    )
    # log(210/100) ≈ 0.742 > ln(2) → sanity error at 2020-06 (first snapshot)
    rows.append(_single_row("11110-b", ppm=210.0, month_ts=months[5]))
    trade_df = pd.DataFrame(rows)
    outliers_df, market_price_df = build_snapshot_outliers(trade_df)

    assert len(outliers_df[outliers_df["month"] == months[5]]) == 0, (
        "First-snapshot-month row must not appear in outliers"
    )
    mkt = market_price_df[
        (market_price_df["aptSeq"] == "11110-b") & (market_price_df["month"] == months[5])
    ]
    assert len(mkt) > 0, "Exempt row should still be counted in market_price_df"


# ---------------------------------------------------------------------------
# Test 2: first snapshot period tracks START_YM via monkeypatch
# ---------------------------------------------------------------------------

def test_first_month_follows_start_ym_monkeypatch(monkeypatch) -> None:
    """2020-06 outlier stays flagged when START_YM is shifted to 2020-03."""
    months = _month_seq("2020-01", 12)
    rows = (
        _flat_rows("11110-anc", 100.0, months)
        + _flat_rows("11110-b", 100.0, months[:5])
        + _flat_rows("11110-b", 100.0, months[6:])
    )
    rows.append(_single_row("11110-b", ppm=210.0, month_ts=months[5]))
    trade_df = pd.DataFrame(rows)

    # With START_YM=202003, 2020-06 is NOT the first snapshot period → not exempt
    monkeypatch.setattr("config.settings.START_YM", "202003")
    outliers_df, _ = build_snapshot_outliers(trade_df)
    assert len(outliers_df[outliers_df["month"] == months[5]]) >= 1, (
        "2020-06 should still be flagged when START_YM=202003"
    )


# ---------------------------------------------------------------------------
# Test 3: age==0 + construction_year>0 + abs_deviation → exempt
# ---------------------------------------------------------------------------

def test_exempt_age_zero(monkeypatch) -> None:
    """age==0 with valid construction_year is exempt regardless of deviation."""
    monkeypatch.setattr("config.settings.START_YM", "201001")
    months = _month_seq("2020-01", 14)
    area = 200.0
    bppm = 2000.0
    rows = (
        _flat_rows("11110-anc", bppm, months, area=area)
        + _flat_rows("11110-b", bppm, months[:12], area=area, age=0, cy=2020)
    )
    # delta=35000 >> 30000 → abs_mask fires; but age=0 cy=2020 → exempt
    rows.append(_single_row(
        "11110-b", price_manwon=int(bppm * area) + 35_000,
        month_ts=months[12], area=area, age=0, cy=2020,
    ))
    trade_df = pd.DataFrame(rows)
    outliers_df, _ = build_snapshot_outliers(trade_df)
    assert len(outliers_df[
        (outliers_df["aptSeq"] == "11110-b") & (outliers_df["month"] == months[12])
    ]) == 0, "age==0 with valid cy should be exempt"


# ---------------------------------------------------------------------------
# Test 4: age=NaN is NOT exempt (연식 미상 guard)
# ---------------------------------------------------------------------------

def test_age_nan_not_exempted(monkeypatch) -> None:
    """age=NaN must not be treated as age==0 exempt."""
    monkeypatch.setattr("config.settings.START_YM", "201001")
    months = _month_seq("2020-01", 14)
    area = 200.0
    bppm = 2000.0
    rows = (
        _flat_rows("11110-anc", bppm, months, area=area)
        + _flat_rows("11110-b", bppm, months[:12], area=area, age=np.nan, cy=0)
    )
    rows.append(_single_row(
        "11110-b", price_manwon=int(bppm * area) + 35_000,
        month_ts=months[12], area=area, age=np.nan, cy=0,
    ))
    trade_df = pd.DataFrame(rows)
    outliers_df, _ = build_snapshot_outliers(trade_df)
    assert len(outliers_df[
        (outliers_df["aptSeq"] == "11110-b") & (outliers_df["month"] == months[12])
    ]) >= 1, "age=NaN must NOT be exempt"


# ---------------------------------------------------------------------------
# Test 5: abs_deviation fires when band_pct not exceeded
# ---------------------------------------------------------------------------

def test_abs_deviation_triggers_outlier(monkeypatch) -> None:
    """abs_deviation reason fires even when relative dev is below band floor."""
    monkeypatch.setattr("config.settings.START_YM", "201001")
    months = _month_seq("2020-01", 14)
    area = 200.0
    bppm = 2000.0
    rows = (
        _flat_rows("11110-anc", bppm, months, area=area)
        + _flat_rows("11110-b", bppm, months[:12], area=area)
    )
    # dev_ratio ≈ 15% < FLOOR_PCT_BASE 18% — no band candidate; abs_dev=30001 ≥ 30000
    rows.append(_single_row(
        "11110-b", price_manwon=int(bppm * area) + 30_001,
        month_ts=months[12], area=area,
    ))
    trade_df = pd.DataFrame(rows)
    outliers_df, _ = build_snapshot_outliers(trade_df)
    b13 = outliers_df[
        (outliers_df["aptSeq"] == "11110-b") & (outliers_df["month"] == months[12])
    ]
    assert len(b13) >= 1
    assert b13.iloc[0]["outlier_reason"] == "abs_deviation"


# ---------------------------------------------------------------------------
# Test 6: sanity_error takes precedence over abs_deviation
# ---------------------------------------------------------------------------

def test_abs_deviation_priority_under_sanity(monkeypatch) -> None:
    """sanity_error has higher priority than abs_deviation in outlier_reason."""
    monkeypatch.setattr("config.settings.START_YM", "201001")
    months = _month_seq("2020-01", 14)
    area = 200.0
    bppm = 2000.0
    rows = (
        _flat_rows("11110-anc", bppm, months, area=area)
        + _flat_rows("11110-b", bppm, months[:12], area=area)
    )
    # log(5000/2000) ≈ 0.916 > ln(2) → sanity_error; also abs_dev ≫ 30000
    rows.append(_single_row("11110-b", ppm=5000.0, month_ts=months[12], area=area))
    trade_df = pd.DataFrame(rows)
    outliers_df, _ = build_snapshot_outliers(trade_df)
    b13 = outliers_df[
        (outliers_df["aptSeq"] == "11110-b") & (outliers_df["month"] == months[12])
    ]
    assert len(b13) >= 1
    assert b13.iloc[0]["outlier_reason"] == "sanity_error"


# ---------------------------------------------------------------------------
# Test 7: exempt row is counted in market_price_df (force-false not drop)
# ---------------------------------------------------------------------------

def test_market_price_includes_exempt(monkeypatch) -> None:
    """Force-false exempt row contributes to market_price_df.trade_count."""
    monkeypatch.setattr("config.settings.START_YM", "202006")
    months = _month_seq("2020-01", 12)
    rows = (
        _flat_rows("11110-anc", 100.0, months)
        + _flat_rows("11110-b", 100.0, months[:5])
        + _flat_rows("11110-b", 100.0, months[6:])
    )
    rows.append(_single_row("11110-b", ppm=210.0, month_ts=months[5]))
    trade_df = pd.DataFrame(rows)
    _, market_price_df = build_snapshot_outliers(trade_df)
    mkt = market_price_df[
        (market_price_df["aptSeq"] == "11110-b") & (market_price_df["month"] == months[5])
    ]
    assert len(mkt) > 0
    assert mkt.iloc[0]["trade_count"] >= 1


# ---------------------------------------------------------------------------
# Test 8: abs_mask boundary is in 만원 units (29999 < 30000 ≤ 30001)
# ---------------------------------------------------------------------------

def test_abs_threshold_unit_is_manwon(monkeypatch) -> None:
    """Boundary: delta=29999 not outlier, delta=30001 is outlier (unit regression guard)."""
    monkeypatch.setattr("config.settings.START_YM", "201001")
    area = 200.0
    bppm = 2000.0

    def _run(delta: int) -> bool:
        months = _month_seq("2020-01", 14)
        rows = (
            _flat_rows("11110-anc", bppm, months, area=area)
            + _flat_rows("11110-b", bppm, months[:12], area=area)
        )
        rows.append(_single_row(
            "11110-b", price_manwon=int(bppm * area) + delta,
            month_ts=months[12], area=area,
        ))
        outliers_df, _ = build_snapshot_outliers(pd.DataFrame(rows))
        return len(outliers_df[
            (outliers_df["aptSeq"] == "11110-b") & (outliers_df["month"] == months[12])
        ]) >= 1

    assert not _run(29_999), "delta=29999 (below threshold) must NOT be flagged"
    assert _run(30_001), "delta=30001 (above threshold) must be flagged"


# ---------------------------------------------------------------------------
# Tests 9-11: renovation buffer — trend_month_robust_band path
# ---------------------------------------------------------------------------

def test_renovation_abs_buffer_unit_is_manwon(monkeypatch) -> None:
    """delta=6000 만원 > RENOVATION_ABS_BUFFER_MANWON(5000) → NOT released by renovation."""
    monkeypatch.setattr("config.settings.START_YM", "201001")
    trade_df, target_month = _build_trend_reno_df(600.0, 6_000)
    outliers_df, _ = build_snapshot_outliers(trade_df)
    target_rows = outliers_df[
        (outliers_df["aptSeq"] == "11110-reno-test")
        & (outliers_df["month"] == target_month)
    ]
    assert len(target_rows) >= 1, (
        "delta=6000 exceeds renovation abs buffer → row must remain in outliers"
    )
    assert target_rows.iloc[0]["renovation_buffer_applied"] == False
    assert target_rows.iloc[0]["reference_type"] == "trend_month_robust_band"


def test_renovation_abs_buffer_within_limit(monkeypatch) -> None:
    """delta=4000 만원 ≤ RENOVATION_ABS_BUFFER_MANWON(5000) + dev≤12% → released."""
    monkeypatch.setattr("config.settings.START_YM", "201001")
    trade_df, target_month = _build_trend_reno_df(400.0, 4_000)
    outliers_df, market_price_df = build_snapshot_outliers(trade_df)
    assert len(outliers_df[
        (outliers_df["aptSeq"] == "11110-reno-test")
        & (outliers_df["month"] == target_month)
    ]) == 0, "delta=4000 within renovation buffer → row must be released"
    mkt = market_price_df[
        (market_price_df["aptSeq"] == "11110-reno-test")
        & (market_price_df["month"] == target_month)
    ]
    assert len(mkt) > 0
    assert mkt.iloc[0]["renovation_buffer_count"] >= 1


def test_renovation_flag_cleared_for_exempt_row(monkeypatch) -> None:
    """Case C: renovation would release + exempt condition → is_outlier=False,
    renovation_buffer_applied=False (exempt takes credit, not renovation)."""
    # target is months[8] = 2020-01 + 8 = 2020-09; set START_YM to that month
    monkeypatch.setattr("config.settings.START_YM", "202009")
    trade_df, target_month = _build_trend_reno_df(400.0, 4_000, start_month="2020-01")
    outliers_df, market_price_df = build_snapshot_outliers(trade_df)

    assert len(outliers_df[
        (outliers_df["aptSeq"] == "11110-reno-test")
        & (outliers_df["month"] == target_month)
    ]) == 0, "Exempt first-month row must not appear in outliers"

    mkt = market_price_df[
        (market_price_df["aptSeq"] == "11110-reno-test")
        & (market_price_df["month"] == target_month)
    ]
    if len(mkt) > 0:
        assert mkt.iloc[0]["renovation_buffer_count"] == 0, (
            "Case C: exempt clears renovation credit — buffer_count must be 0"
        )


# ---------------------------------------------------------------------------
# Test 12: missing age column does not crash
# ---------------------------------------------------------------------------

def test_missing_age_column_does_not_crash(monkeypatch) -> None:
    """Pipeline completes without error when input has no age column."""
    monkeypatch.setattr("config.settings.START_YM", "201001")
    months = _month_seq("2020-01", 8)
    rows = _flat_rows("11110-anc", 100.0, months) + _flat_rows("11110-b", 100.0, months)
    trade_df = pd.DataFrame(rows).drop(columns=["age"])
    outliers_df, market_price_df = build_snapshot_outliers(trade_df)
    assert isinstance(outliers_df, pd.DataFrame)
    assert isinstance(market_price_df, pd.DataFrame)
