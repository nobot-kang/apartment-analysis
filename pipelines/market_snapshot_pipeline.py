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

    # Rolling 이동평균 (전체 집계 기준)
    all_monthly = result[result["sggCd"] == "ALL"].set_index("month")["price_median_m2"]
    rolling_map = {}
    for window, col in [(3, "rolling_3m_median_m2"), (6, "rolling_6m_median_m2"), (12, "rolling_12m_median_m2")]:
        roll = all_monthly.sort_index().rolling(window, min_periods=1).mean()
        rolling_map[col] = roll

    def _get_rolling(month: pd.Timestamp, col: str) -> float:
        s = rolling_map[col]
        return float(s.loc[month]) if month in s.index else np.nan

    result["rolling_3m_median_m2"] = result["month"].map(
        lambda m: _get_rolling(m, "rolling_3m_median_m2")
    )
    result["rolling_6m_median_m2"] = result["month"].map(
        lambda m: _get_rolling(m, "rolling_6m_median_m2")
    )
    result["rolling_12m_median_m2"] = result["month"].map(
        lambda m: _get_rolling(m, "rolling_12m_median_m2")
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
# A-3: 이상치·오류·비정상 거래 탐지
# ---------------------------------------------------------------------------

def _compute_monthly_band_frame(df: pd.DataFrame) -> pd.DataFrame:
    """단지×면적×월 대표가격과 moving-average band 를 계산한다."""
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


def build_snapshot_outliers(trade_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A-3용 이상치 탐지 및 단지별 월별 시세 테이블을 생성한다.

    탐지 기준:
        그룹 단위 : aptSeq × area_repr
        기준 시세 : 직전 관측월들의 trailing moving average
        band      : max(rolling std × 2, moving average × 25%)
        판정 방식 :
            1. band 밖 거래는 우선 candidate 로 본다.
            2. 같은 방향 거래가 이후 2개월 이상, 총 3건 이상 이어지면
               일시적 이상치가 아니라 추세 전환으로 간주해 복원한다.
            3. 추세 전환으로 인정된 월 안에서는 월별 중앙값 기준 robust band 로
               개별 row 만 다시 점검한다.
        제외 대상 : 1층 거래
    """
    logger.info(
        "A-3 이상치 탐지 시작 "
        f"(MA {BOLLINGER_WINDOW_MONTHS}개월 + Bollinger band, 최소 폭 {OUTLIER_THRESHOLD*100:.0f}%)..."
    )

    df = trade_df.dropna(subset=["date", "price_per_m2", "area_repr", "aptSeq"]).copy()
    df["month"] = pd.to_datetime(df["month"])

    floor_numeric = pd.to_numeric(df["floor"], errors="coerce")
    n_before = len(df)
    df = df[floor_numeric != 1].reset_index(drop=True)
    logger.info(f"  1층 거래 제외: {n_before - len(df):,}건 → {len(df):,}건")

    logger.info("  단지·면적·월 기준 moving-average band 계산 중...")
    monthly = _compute_monthly_band_frame(df)
    monthly = _annotate_trend_confirmation(monthly)

    merge_cols = [
        "aptSeq",
        "area_repr",
        "month",
        "month_price_m2",
        "month_trade_count",
        "ref_month",
        "ref_price",
        "band_width_abs",
        "band_lower",
        "band_upper",
        "band_width_pct",
        "month_row_band_abs",
        "trend_confirmed",
        "trend_support_months",
        "trend_total_trades",
        "trend_ref_price",
    ]
    evaluated = df.merge(monthly[merge_cols], on=["aptSeq", "area_repr", "month"], how="left")

    band_outlier = (
        evaluated["ref_price"].notna()
        & ~evaluated["trend_confirmed"].fillna(False)
        & (
            evaluated["price_per_m2"].lt(evaluated["band_lower"])
            | evaluated["price_per_m2"].gt(evaluated["band_upper"])
        )
    )
    trend_row_outlier = (
        evaluated["trend_confirmed"].fillna(False)
        & evaluated["month_trade_count"].fillna(0).ge(TREND_ROW_MIN_TRADE_COUNT)
        & evaluated["month_price_m2"].notna()
        & (evaluated["price_per_m2"] - evaluated["month_price_m2"]).abs().gt(evaluated["month_row_band_abs"])
    )

    evaluated["is_outlier"] = band_outlier | trend_row_outlier
    evaluated["reference_type"] = np.where(
        trend_row_outlier,
        "trend_month_robust_band",
        "moving_average_band",
    )
    evaluated["effective_ref_price"] = evaluated["ref_price"]
    evaluated["effective_ref_month"] = evaluated["ref_month"]
    evaluated["effective_band_width_abs"] = evaluated["band_width_abs"]
    evaluated.loc[trend_row_outlier, "effective_ref_price"] = evaluated.loc[trend_row_outlier, "month_price_m2"]
    evaluated.loc[trend_row_outlier, "effective_ref_month"] = evaluated.loc[trend_row_outlier, "month"]
    evaluated.loc[trend_row_outlier, "effective_band_width_abs"] = evaluated.loc[trend_row_outlier, "month_row_band_abs"]
    evaluated["effective_band_width_pct"] = np.where(
        evaluated["effective_ref_price"] > 0,
        evaluated["effective_band_width_abs"] / evaluated["effective_ref_price"] * 100,
        np.nan,
    )
    evaluated["price_deviation_pct"] = np.where(
        evaluated["effective_ref_price"] > 0,
        (evaluated["price_per_m2"] - evaluated["effective_ref_price"]) / evaluated["effective_ref_price"] * 100,
        np.nan,
    )
    evaluated["outlier_direction"] = pd.Series(pd.NA, index=evaluated.index, dtype="object")
    has_deviation = evaluated["price_deviation_pct"].notna()
    evaluated.loc[has_deviation & evaluated["price_deviation_pct"].gt(0), "outlier_direction"] = "고가이상치"
    evaluated.loc[has_deviation & evaluated["price_deviation_pct"].le(0), "outlier_direction"] = "저가이상치"

    outliers_df = evaluated[evaluated["is_outlier"]].copy()
    outliers_df["ref_month"] = outliers_df["effective_ref_month"]
    outliers_df["ref_price"] = outliers_df["effective_ref_price"]
    outliers_df["band_width_pct"] = outliers_df["effective_band_width_pct"]
    keep_cols = [
        "month",
        "date",
        "aptSeq",
        "apt_name",
        "dong",
        "dong_repr",
        "area",
        "area_repr",
        "floor",
        "construction_year",
        "age",
        "price",
        "price_per_m2",
        "ref_month",
        "ref_price",
        "band_width_pct",
        "price_deviation_pct",
        "outlier_direction",
        "reference_type",
        "trend_confirmed",
        "trend_support_months",
        "trend_total_trades",
        "trend_ref_price",
    ]
    keep_cols = [c for c in keep_cols if c in outliers_df.columns]
    outliers_df = outliers_df[keep_cols].sort_values(["aptSeq", "area_repr", "month", "date"]).reset_index(drop=True)

    market_price_df = (
        evaluated[~evaluated["is_outlier"]]
        .groupby(["aptSeq", "area_repr", "month"], observed=True)["price_per_m2"]
        .median()
        .reset_index()
        .rename(columns={"price_per_m2": "market_price_m2"})
        .sort_values(["aptSeq", "area_repr", "month"])
        .reset_index(drop=True)
    )
    market_price_df["month"] = pd.to_datetime(market_price_df["month"])

    n_total = len(evaluated)
    n_outlier = len(outliers_df)
    trend_seq_count = int(monthly["trend_confirmed"].sum())
    logger.info(
        f"A-3 완료: 이상치 {n_outlier:,}건 / {n_total:,}건 "
        f"({n_outlier / n_total * 100:.2f}%), "
        f"추세 전환 월 {trend_seq_count:,}개, "
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
