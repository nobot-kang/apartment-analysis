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

# 이상치 탐지: 직전 시세 대비 편차 임계값 (25%)
OUTLIER_THRESHOLD: float = 0.25
# 시세 조회 최대 소급 개월 수
LOOKBACK_MONTHS: int = 6
# 사전 필터: 동일 단지·면적 내 시간순 직전 거래 대비 등락 임계값 (25%)
# 이 임계값을 초과하는 거래는 시세 계산·이상치 탐지 양쪽에서 모두 제외
CONSECUTIVE_THRESHOLD: float = 0.25


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
# A-3: 이상치·오류·비정상 거래 탐지 (단지·면적 기준, 시세 대비 ±20%)
# ---------------------------------------------------------------------------

def build_snapshot_outliers(
    trade_df: pd.DataFrame,
    n_iterations: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A-3용 이상치 탐지 및 단지별 월별 시세 테이블을 생성한다.

    탐지 기준:
        그룹 단위 : aptSeq × area_repr (같은 단지 내 floor(전용면적) 동일)
        참조 시세 : 해당 거래월 직전 최대 6개월 이내 가장 최근 월의 중앙값
        이상치 조건: |거래가 - 참조 시세| / 참조 시세 > 20%
        참조 없음  : 직전 6개월 내 거래 이력 없으면 탐지 생략
        1층 거래   : 시세 계산 및 이상치 탐지 모두 제외

    알고리즘 (양방향 반복 수렴):
        1. 1층 거래 제외
        2. n_iterations 회 반복:
           a. 현재 이상치 집합을 제외한 거래로 월별 중앙값(ref_price) 계산
              → 이상치가 복원되면 다음 반복에서 중앙값에 즉시 반영
           b. merge_asof 로 전체 거래(이상치 포함)에 ref_price 매핑
              (당월 제외, 최대 6개월 소급)
           c. ±20% 초과 → 새 이상치 집합 산출 (양방향: 추가 + 복원 모두 반영)
           d. 이상치 집합이 변화 없으면 조기 수렴 종료
        3. 최종 이상치 완전 제외 후 클린 시세(market_price_df) 산출
        4. 클린 시세로 최종 merge_asof 재수행 → 이상치 출력의 ref_price를
           가장 정제된 기준으로 업데이트

    Args:
        trade_df    : 공통 전처리가 완료된 매매 DataFrame
        n_iterations: 최대 반복 횟수 (기본 5회, 통상 3~4회차에 수렴)

    Returns:
        outliers_df     : 이상치로 분류된 거래 행 + 탐지 메타 컬럼
                          (ref_price·price_deviation_pct 은 최종 클린 시세 기준)
        market_price_df : 이상치 제외 최종 단지·면적별 월별 시세 (만원/㎡)
    """
    logger.info(
        f"A-3 이상치 탐지 시작 (단지·면적 기준, ±{OUTLIER_THRESHOLD*100:.0f}%, "
        f"최대 {n_iterations}회 반복)..."
    )

    df = (
        trade_df
        .dropna(subset=["date", "price_per_m2", "area_repr"])
        .copy()
    )
    df["month"] = pd.to_datetime(df["month"])

    # ------------------------------------------------------------------
    # 사전 필터 ① 1층 거래 제외 – 저가 경향이 강해 시세 기준에서 제외
    # ------------------------------------------------------------------
    floor_numeric = pd.to_numeric(df["floor"], errors="coerce")
    n_before = len(df)
    df = df[floor_numeric != 1].reset_index(drop=True)
    logger.info(
        f"  사전 필터 ① 1층 거래 제외: {n_before - len(df):,}건 → {len(df):,}건"
    )

    # ------------------------------------------------------------------
    # 사전 필터 ② 직전 유효 거래 대비 30% 초과 등락 거래 제외 (순차 비교)
    #   - 그룹: (aptSeq, area_repr), 정렬 기준: date (시간순)
    #   - 이상 거래가 제거되면 last_valid_price 를 갱신하지 않고
    #     다음 거래는 그 이전 유효 거래와 비교 → 연쇄 제거 방지
    #     예) A(100) → B(200, 제거) → C(110): C 는 A 기준 +10% → 유지
    # ------------------------------------------------------------------
    def _sequential_consec_mask(prices: np.ndarray, threshold: float) -> np.ndarray:
        """시간순 정렬된 price_per_m2 배열에서 연쇄 제거 없이 이상 거래를 마킹."""
        mask = np.zeros(len(prices), dtype=bool)
        if len(prices) == 0:
            return mask
        last_valid = float(prices[0])
        for i in range(1, len(prices)):
            p = float(prices[i])
            if last_valid <= 0:
                # 유효하지 않은 기준가 → 현재 거래로 기준 교체
                last_valid = p
                continue
            if abs(p - last_valid) / last_valid > threshold:
                mask[i] = True          # 이상 거래 마킹, last_valid 갱신 안 함
            else:
                last_valid = p          # 정상 거래 → 다음 비교 기준 갱신
        return mask

    df_tmp = df.sort_values(["aptSeq", "area_repr", "date"])
    consec_exclude_mask = (
        df_tmp
        .groupby(["aptSeq", "area_repr"], group_keys=False)[["price_per_m2"]]
        .apply(
            lambda g: pd.Series(
                _sequential_consec_mask(g["price_per_m2"].values, CONSECUTIVE_THRESHOLD),
                index=g.index,
            )
        )
        .squeeze()
    )
    consec_exclude_idx = consec_exclude_mask[consec_exclude_mask].index

    n_before = len(df)
    df = df.drop(index=consec_exclude_idx).reset_index(drop=True)
    logger.info(
        f"  사전 필터 ② 연속 등락 {CONSECUTIVE_THRESHOLD*100:.0f}% 초과 제외 (순차): "
        f"{n_before - len(df):,}건 → {len(df):,}건"
    )

    # 원본 행 위치를 반복 간 추적하기 위한 컬럼
    df["_orig_idx"] = range(len(df))

    tolerance = pd.Timedelta(days=LOOKBACK_MONTHS * 31)

    # 이상치 마스크 초기화 (첫 반복은 이상치 없음 → 전체 거래로 중앙값 계산)
    is_outlier = pd.Series(False, index=range(len(df)), dtype=bool)

    for iteration in range(n_iterations):
        # ------------------------------------------------------------------
        # Step a: 현재 이상치 집합을 제외한 클린 거래로 월별 평균값(이동평균) 계산
        #   - 이전 반복에서 복원된 거래(이상치 → 정상)도 여기서 평균값에 반영됨
        # ------------------------------------------------------------------
        monthly_med = (
            df[~is_outlier]
            .groupby(["aptSeq", "area_repr", "month"])["price_per_m2"]
            .mean()
            .reset_index()
            .rename(columns={"price_per_m2": "ref_price", "month": "ref_month"})
        )
        monthly_med["ref_month"] = pd.to_datetime(monthly_med["ref_month"])
        monthly_med = monthly_med.sort_values("ref_month").reset_index(drop=True)

        # ------------------------------------------------------------------
        # Step b: 전체 거래(이상치 포함)에 ref_price 매핑
        #   _key = month - 1초: 당월 중앙값을 참조에서 배제
        #   이상치 거래도 포함하여 매핑 → 복원 여부 판단 가능
        # ------------------------------------------------------------------
        df_sorted = df.sort_values("month").copy()
        df_sorted["_key"] = df_sorted["month"] - pd.Timedelta(seconds=1)

        merged = pd.merge_asof(
            df_sorted,
            monthly_med,
            left_on="_key",
            right_on="ref_month",
            by=["aptSeq", "area_repr"],
            direction="backward",
            tolerance=tolerance,
        )
        merged.drop(columns=["_key"], inplace=True)

        # ------------------------------------------------------------------
        # Step c: 새 이상치 집합 산출 (양방향)
        #   - 기존 이상치 중 ±20% 이내로 복귀한 거래 → 복원
        #   - 기존 정상 거래 중 ±20% 초과로 변한 거래 → 신규 편입
        # ------------------------------------------------------------------
        has_ref = merged["ref_price"].notna()
        merged["price_deviation_pct"] = np.where(
            has_ref,
            (merged["price_per_m2"] - merged["ref_price"]) / merged["ref_price"] * 100,
            np.nan,
        )
        new_outlier_flag = has_ref & (
            merged["price_deviation_pct"].abs() > OUTLIER_THRESHOLD * 100
        )

        # _orig_idx 기준으로 이상치 마스크 재구성 (전체 교체 → 복원 자동 반영)
        outlier_orig_idx = set(merged.loc[new_outlier_flag, "_orig_idx"].values)
        new_is_outlier = pd.Series(
            [idx in outlier_orig_idx for idx in range(len(df))],
            dtype=bool,
        )

        # 양방향 변화량 로깅
        prev_set = set(df.index[is_outlier])
        new_set  = set(df.index[new_is_outlier])
        added    = len(new_set - prev_set)
        restored = len(prev_set - new_set)
        logger.info(
            f"  반복 {iteration + 1}/{n_iterations}: "
            f"총 {int(is_outlier.sum()):,} → {int(new_is_outlier.sum()):,}건 "
            f"(+{added:,} 추가 / -{restored:,} 복원)"
        )

        # ------------------------------------------------------------------
        # Step d: 수렴 확인 후 이상치 집합 교체
        # ------------------------------------------------------------------
        converged = new_is_outlier.equals(is_outlier)
        is_outlier = new_is_outlier   # 복원된 거래는 다음 반복 중앙값에 반영됨

        if converged:
            logger.info(f"  ✓ 수렴: {iteration + 1}회 반복 후 종료")
            break

    # ------------------------------------------------------------------
    # 최종 클린 시세 – 수렴된 이상치 완전 제외 후 재계산
    # ------------------------------------------------------------------
    logger.info("  최종 시세 계산 (수렴된 이상치 완전 제외, 월별 평균값)...")
    market_price_df = (
        df[~is_outlier]
        .groupby(["aptSeq", "area_repr", "month"])["price_per_m2"]
        .mean()
        .reset_index()
        .rename(columns={"price_per_m2": "market_price_m2"})
    )
    market_price_df["month"] = pd.to_datetime(market_price_df["month"])

    # ------------------------------------------------------------------
    # 이상치 출력용 ref_price 최종 갱신
    #   반복 중 사용한 중간 중앙값이 아니라, 완전히 정제된 market_price_df 로
    #   이상치 거래의 ref_price · price_deviation_pct 를 재계산한다.
    # ------------------------------------------------------------------
    logger.info("  이상치 ref_price 최종 갱신 (클린 시세 기준)...")
    clean_ref = (
        market_price_df
        .rename(columns={"market_price_m2": "ref_price", "month": "ref_month"})
        .sort_values("ref_month")
    )

    outlier_rows = df[is_outlier].sort_values("month").copy()
    outlier_rows["_key"] = outlier_rows["month"] - pd.Timedelta(seconds=1)

    outlier_with_final_ref = pd.merge_asof(
        outlier_rows,
        clean_ref,
        left_on="_key",
        right_on="ref_month",
        by=["aptSeq", "area_repr"],
        direction="backward",
        tolerance=tolerance,
    )
    outlier_with_final_ref.drop(columns=["_key"], inplace=True)

    # 최종 클린 시세 기준 편차 재계산
    has_final_ref = outlier_with_final_ref["ref_price"].notna()
    outlier_with_final_ref["price_deviation_pct"] = np.where(
        has_final_ref,
        (outlier_with_final_ref["price_per_m2"] - outlier_with_final_ref["ref_price"])
        / outlier_with_final_ref["ref_price"] * 100,
        np.nan,
    )
    outlier_with_final_ref["outlier_direction"] = np.where(
        outlier_with_final_ref["price_deviation_pct"] > 0, "고가이상치", "저가이상치"
    )

    keep_cols = [
        "month", "date",
        "aptSeq", "apt_name", "dong", "dong_repr",
        "area", "area_repr", "floor", "construction_year", "age",
        "price", "price_per_m2",
        "ref_month", "ref_price", "price_deviation_pct", "outlier_direction",
    ]
    keep_cols = [c for c in keep_cols if c in outlier_with_final_ref.columns]

    outliers_df = (
        outlier_with_final_ref[keep_cols]
        .sort_values(["aptSeq", "area_repr", "month"])
        .reset_index(drop=True)
    )

    # 임시 컬럼 정리
    df.drop(columns=["_orig_idx"], inplace=True)

    n_total   = len(df)
    n_outlier = len(outliers_df)
    logger.info(
        f"A-3 완료: 이상치 {n_outlier:,}건 / {n_total:,}건 "
        f"({n_outlier / n_total * 100:.2f}%), "
        f"최종 시세 {len(market_price_df):,}행 (단지×면적×월)"
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
