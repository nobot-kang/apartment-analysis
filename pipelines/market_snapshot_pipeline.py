"""Section A – 데이터 진단 & 시장 스냅샷 집계 파이프라인.

기존 processed parquet (apt_trade_*.parquet, apt_rent_*.parquet)을 로드하여
대시보드 Section A에서 필요한 집계 parquet 3종을 생성한다.

출력 위치: data/preprocessed_plus/
  - snapshot_monthly_trade.parquet  : A-1 월별 거래량·가격 추이 (매매)
  - snapshot_monthly_rent.parquet   : A-1 월별 거래량·가격 추이 (전월세)
  - snapshot_area_mix.parquet       : A-2 면적 믹스 변화 & 구성효과 분해
  - snapshot_outliers.parquet       : A-3 이상치·비정상 거래 탐지 결과
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

import sys
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import PROCESSED_DIR, ALL_REGIONS, SEOUL_REGIONS

# 출력 디렉토리
PREPROCESSED_PLUS_DIR: Path = _project_root / "data" / "preprocessed_plus"

# 면적 구간 정의 (㎡)
AREA_BUCKETS = [
    (0,   60,   "~60㎡"),
    (60,  85,   "60~85㎡"),
    (85,  102,  "85~102㎡"),
    (102, 9999, "102㎡~"),
]

# 이상치 탐지: moving average band 의 최소 상대 폭
OUTLIER_THRESHOLD: float = 0.25
# 시세 조회 최대 소급 개월 수
LOOKBACK_MONTHS: int = 6
# Bollinger band 파라미터
BOLLINGER_WINDOW_MONTHS: int = 6
BOLLINGER_MIN_HISTORY_MONTHS: int = 3
BOLLINGER_STD_MULTIPLIER: float = 2.0
# 급격한 가격 이동이 추세 전환인지 확인하는 파라미터
TREND_LOOKAHEAD_MONTHS: int = 6
TREND_MIN_SUPPORT_MONTHS: int = 2
TREND_MIN_TOTAL_TRADES: int = 3
TREND_SUPPORT_BAND_RATIO: float = 0.5
TREND_ALIGNMENT_TOLERANCE: float = 0.12
# 추세 전환으로 인정된 월 안에서 개별 행을 다시 점검할 때 쓰는 band
TREND_ROW_MIN_TRADE_COUNT: int = 3
TREND_ROW_STD_MULTIPLIER: float = 2.5
TREND_ROW_MIN_BAND_PCT: float = 0.08

# A-3 v2: spread-based outlier detection
PATH_WINDOW_MONTHS: int = 7
PATH_MIN_PERIODS: int = 3
SPREAD_WINDOW_MONTHS: int = 9
SHRINK_K: int = 6
BAND_Z: float = 3.0
FLOOR_PCT_BASE: float = 0.18
FLOOR_PCT_SPARSE_ADDON: float = 0.05
SANITY_LOG_RATIO: float = 0.693          # ln(2.0)
LEADER_SPREAD_MONTHS: int = 24
LEADER_SPREAD_SIGN_RATIO: float = 0.80
OLD_COMPLEX_AGE: int = 20
RENOVATION_ABS_BUFFER_KRW: int = 50_000_000
RENOVATION_REL_CAP: float = 0.12


def _load_all_trade(processed_dir: Path) -> pd.DataFrame:
    """모든 apt_trade_*.parquet을 로드해 통합한다."""
    files = sorted(processed_dir.glob("apt_trade_*.parquet"))
    if not files:
        logger.warning("apt_trade parquet 파일이 없습니다.")
        return pd.DataFrame()

    dfs = []
    for f in tqdm(files, desc="Loading trade parquets"):
        df = pd.read_parquet(f)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _load_all_rent(processed_dir: Path) -> pd.DataFrame:
    """모든 apt_rent_*.parquet을 로드해 통합한다."""
    files = sorted(processed_dir.glob("apt_rent_*.parquet"))
    if not files:
        logger.warning("apt_rent parquet 파일이 없습니다.")
        return pd.DataFrame()

    dfs = []
    for f in tqdm(files, desc="Loading rent parquets"):
        df = pd.read_parquet(f)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _add_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    """aptSeq 컬럼에서 sggCd와 region_name, region_type을 파생한다.

    aptSeq 형식: "{sggCd}-{complexId}" (예: "11110-42")
    """
    df = df.copy()
    df["sggCd"] = df["aptSeq"].astype(str).str.split("-").str[0]
    df["region_name"] = df["sggCd"].map(ALL_REGIONS).fillna("기타")
    seoul_codes = set(SEOUL_REGIONS.keys())
    df["region_type"] = df["sggCd"].apply(
        lambda c: "서울" if c in seoul_codes else "경기"
    )
    return df


def _add_area_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """area 컬럼으로 면적 구간(area_bucket)을 생성한다."""
    df = df.copy()
    conditions = [
        (df["area"] < 60),
        (df["area"] >= 60) & (df["area"] < 85),
        (df["area"] >= 85) & (df["area"] < 102),
        (df["area"] >= 102),
    ]
    choices = [label for _, _, label in AREA_BUCKETS]
    df["area_bucket"] = np.select(conditions, choices, default="기타")
    return df


def _add_month_column(df: pd.DataFrame) -> pd.DataFrame:
    """date 컬럼에서 월 시작일(month)을 생성한다."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


# ---------------------------------------------------------------------------
# A-1: 월별 거래량·중위 ㎡당 가격·분산 추이
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# A-2: 면적 믹스 변화 & 구성효과 분해
# ---------------------------------------------------------------------------

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
    all_df = df[df["sggCd"] == "ALL"].copy() if "ALL" in df["sggCd"].values else df.copy()
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


# ---------------------------------------------------------------------------
# A-3: 이상치·오류·비정상 거래 탐지 (v2: spread-based)
# ---------------------------------------------------------------------------

def _centered_rolling_median(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    result = s.rolling(window, center=True, min_periods=min_periods).median()
    result = result.ffill().bfill()
    return result


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


def _build_cohort_paths(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C1 (sggCd × area_bucket × month) 및 C2 (sggCd × month) 코호트 가격 경로를 계산한다."""
    def _smooth_group(grp: pd.DataFrame, val_col: str, out_col: str) -> pd.DataFrame:
        grp = grp.sort_values("month").copy()
        log_vals = np.where(grp[val_col] > 0, np.log(grp[val_col]), np.nan)
        smoothed = _centered_rolling_median(pd.Series(log_vals, index=grp.index), PATH_WINDOW_MONTHS, PATH_MIN_PERIODS)
        grp[out_col] = np.exp(smoothed)
        return grp

    # C1: sggCd × area_bucket × month
    c1_raw = (
        df.groupby(["sggCd", "area_bucket", "month"], observed=True)["price_per_m2"]
        .median()
        .reset_index()
        .rename(columns={"price_per_m2": "_med"})
    )
    c1_parts = []
    for keys, grp in c1_raw.groupby(["sggCd", "area_bucket"], observed=True):
        c1_parts.append(_smooth_group(grp, "_med", "path_c1_m2"))
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
        c2_parts.append(_smooth_group(grp, "_med", "path_c2_m2"))
    c2_df = pd.concat(c2_parts, ignore_index=True)[["sggCd", "month", "path_c2_m2"]]

    return c1_df, c2_df


def _build_complex_spreads(
    df: pd.DataFrame, c1_df: pd.DataFrame, c2_df: pd.DataFrame
) -> pd.DataFrame:
    """단지별 G0/G1/G2 스프레드 경로와 수축(shrinkage) 혼합 기준가를 계산한다."""

    def _smooth_group_col(sub: pd.DataFrame, val_col: str, out_col: str) -> pd.DataFrame:
        sub = sub.sort_values("month").copy()
        log_vals = np.where(sub[val_col] > 0, np.log(sub[val_col]), np.nan)
        smoothed = _centered_rolling_median(pd.Series(log_vals, index=sub.index), PATH_WINDOW_MONTHS, PATH_MIN_PERIODS)
        sub[out_col] = np.exp(smoothed)
        return sub

    # G0: aptSeq × area_repr × month
    g0_raw = (
        df.groupby(["aptSeq", "area_repr", "month"], observed=True)["price_per_m2"]
        .median()
        .reset_index()
        .rename(columns={"price_per_m2": "_g0_med"})
    )
    g0_parts = []
    for keys, grp in g0_raw.groupby(["aptSeq", "area_repr"], observed=True):
        g0_parts.append(_smooth_group_col(grp, "_g0_med", "path_g0_m2"))
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
        g1_parts.append(_smooth_group_col(grp, "_g1_med", "path_g1_m2"))
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
        g2_parts.append(_smooth_group_col(grp, "_g2_med", "path_g2_m2"))
    g2_df = pd.concat(g2_parts, ignore_index=True)[["aptSeq", "month", "path_g2_m2"]]

    # Merge sggCd and area_bucket onto G0 frame (take first value per aptSeq)
    meta = df[["aptSeq", "area_repr", "sggCd", "area_bucket"]].drop_duplicates(["aptSeq", "area_repr"])
    frame = g0_df.merge(meta, on=["aptSeq", "area_repr"], how="left")

    # Merge G1
    frame = frame.merge(g1_df, on=["aptSeq", "area_bucket", "month"], how="left")
    # Merge G2
    frame = frame.merge(g2_df, on=["aptSeq", "month"], how="left")
    # Merge C1
    frame = frame.merge(c1_df, on=["sggCd", "area_bucket", "month"], how="left")
    # Merge C2
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
        g0 = sub["_path_g0_m2"].dropna()
        ref = sub["ref_price_m2"].dropna()
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

    # Build is_outlier from spread-based logic
    is_outlier_v2 = sanity_mask | unsupported_mask
    outlier_reason_v2 = np.where(
        sanity_mask, "sanity_error",
        np.where(unsupported_mask, "unsupported_jump", ""),
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
    age_arr = pd.to_numeric(evaluated.get("age", pd.Series(0, index=evaluated.index)), errors="coerce").fillna(0)
    ref_total = ref_m2 * evaluated["area"].to_numpy(dtype=float)
    price_arr = evaluated["price"].to_numpy(dtype=float)
    dev_pct_arr = np.where(ref_m2 > 0, (ppm - ref_m2) / ref_m2 * 100.0, np.nan)

    reno_mask = (
        is_outlier_final
        & (ppm > ref_m2)
        & (age_arr.to_numpy() >= OLD_COMPLEX_AGE)
        & ((price_arr - ref_total) <= RENOVATION_ABS_BUFFER_KRW)
        & (dev_pct_arr <= RENOVATION_REL_CAP * 100)
    )
    evaluated.loc[reno_mask, "is_outlier"] = False
    evaluated.loc[reno_mask, "renovation_buffer_applied"] = True

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
    n_reno = int(np.sum(reno_mask))
    n_trend_row = int(trend_row_outlier.sum())
    logger.info(
        f"A-3 v2 완료: 이상치 {n_outlier:,}건 / {n_total:,}건 "
        f"({n_outlier / n_total * 100:.2f}%) | "
        f"sanity_error={n_sanity}, unsupported_jump={n_unsupported}, "
        f"trend_month_robust_band={n_trend_row}, renovation_buffer={n_reno}, "
        f"최종 시세 {len(market_price_df):,}행"
    )
    return outliers_df, market_price_df


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------

class MarketSnapshotPipeline:
    """Section A 시장 스냅샷 집계 파이프라인."""

    def __init__(
        self,
        processed_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.output_dir = output_dir or PREPROCESSED_PLUS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, df: pd.DataFrame, filename: str) -> None:
        """parquet으로 저장하고 파일 크기를 확인한다."""
        if df.empty:
            logger.warning(f"저장 스킵 (빈 DataFrame): {filename}")
            return
        out_path = self.output_dir / filename
        df.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info(f"저장 완료: {filename} ({len(df):,}행, {size_mb:.1f} MB)")
        if size_mb > 90:
            logger.warning(f"파일 크기 주의: {filename} ({size_mb:.1f} MB) – GitHub LFS 한도(100MB) 근접")

    def run(self) -> None:
        """전체 파이프라인을 실행한다."""
        logger.info("=== MarketSnapshotPipeline 시작 ===")

        # 데이터 로딩
        trade_raw = _load_all_trade(self.processed_dir)
        rent_raw = _load_all_rent(self.processed_dir)

        if trade_raw.empty:
            logger.error("매매 데이터가 없습니다. 파이프라인을 중단합니다.")
            return

        logger.info(f"매매 로드 완료: {len(trade_raw):,}건")
        logger.info(f"전월세 로드 완료: {len(rent_raw):,}건")

        # 공통 전처리
        logger.info("공통 전처리 (region, area_bucket, month 컬럼 추가)...")
        trade_df = _add_region_columns(trade_raw)
        trade_df = _add_area_bucket(trade_df)
        trade_df = _add_month_column(trade_df)

        if not rent_raw.empty:
            rent_df = _add_region_columns(rent_raw)
            rent_df = _add_area_bucket(rent_df)
            rent_df = _add_month_column(rent_df)
        else:
            rent_df = pd.DataFrame()

        # A-1: 월별 집계
        monthly_trade = build_snapshot_monthly_trade(trade_df)
        self._save(monthly_trade, "snapshot_monthly_trade.parquet")

        if not rent_df.empty:
            monthly_rent = build_snapshot_monthly_rent(rent_df)
            self._save(monthly_rent, "snapshot_monthly_rent.parquet")

        # A-2: 면적 믹스
        area_mix = build_snapshot_area_mix(trade_df)
        self._save(area_mix, "snapshot_area_mix.parquet")

        # A-3: 이상치 탐지 + 단지 시세
        outliers, market_price = build_snapshot_outliers(trade_df)
        self._save(outliers, "snapshot_outliers.parquet")
        self._save(market_price, "snapshot_complex_market_price.parquet")

        logger.info("=== MarketSnapshotPipeline 완료 ===")


if __name__ == "__main__":
    pipeline = MarketSnapshotPipeline()
    pipeline.run()
