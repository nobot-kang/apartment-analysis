"""분석 전반에서 재사용하는 공통 로더와 집계 유틸리티."""

from __future__ import annotations

import importlib
from typing import Sequence

import numpy as np
import pandas as pd

from config.settings import ALL_REGIONS, GYEONGGI_REGIONS, PROCESSED_DIR, SEOUL_REGIONS

ANALYSIS_START_YM = "202001"
DASHBOARD_START_YM = ANALYSIS_START_YM
DASHBOARD_START_YEAR = int(DASHBOARD_START_YM[:4])
AREA_BINS = [0.0, 60.0, 85.0, np.inf]
AREA_LABELS = ["60㎡ 이하", "60~85㎡", "85㎡ 초과"]
FLOOR_BINS = [0.0, 5.0, 15.0, np.inf]
FLOOR_LABELS = ["저층(1~5층)", "중층(6~15층)", "고층(16층+)"]
AGE_LABELS = ["신축(0~5년)", "준신축(6~15년)", "구축(16~30년)", "노후(30년+)"]

TRADE_WEIGHTED_COLUMNS = [
    "평균거래금액",
    "중앙값거래금액",
    "평균84환산금액",
    "중앙값84환산금액",
    "절사평균거래금액",
    "평균전용면적",
    "평균건물연령",
    "평균거래금액_60㎡이하",
    "평균거래금액_60~85㎡",
    "평균거래금액_85㎡초과",
]
RENT_WEIGHTED_COLUMNS = [
    "평균보증금",
    "중앙값보증금",
    "평균월세",
    "중앙값월세",
    "평균84환산보증금",
]

POLICY_EVENTS = {
    "2020-06-01": "6.17 대책",
    "2021-08-01": "금리 인상 시작",
    "2022-10-01": "레고랜드 자금경색",
    "2023-01-01": "특례보금자리론",
}

SEOUL_DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "11110": (37.5730, 126.9794),
    "11140": (37.5636, 126.9976),
    "11170": (37.5326, 126.9905),
    "11200": (37.5633, 127.0369),
    "11215": (37.5385, 127.0823),
    "11230": (37.5744, 127.0396),
    "11260": (37.6063, 127.0927),
    "11290": (37.5894, 127.0167),
    "11305": (37.6396, 127.0257),
    "11320": (37.6688, 127.0471),
    "11350": (37.6542, 127.0568),
    "11380": (37.6176, 126.9227),
    "11410": (37.5794, 126.9368),
    "11440": (37.5663, 126.9019),
    "11470": (37.5170, 126.8665),
    "11500": (37.5509, 126.8495),
    "11530": (37.4955, 126.8875),
    "11545": (37.4569, 126.8956),
    "11560": (37.5264, 126.8962),
    "11590": (37.5124, 126.9393),
    "11620": (37.4784, 126.9516),
    "11650": (37.4837, 127.0324),
    "11680": (37.5172, 127.0473),
    "11710": (37.5145, 127.1059),
    "11740": (37.5301, 127.1238),
}

DASHBOARD_TRADE_SUMMARY_PATH = PROCESSED_DIR / "dashboard_trade_summary.parquet"
DASHBOARD_RENT_SUMMARY_PATH = PROCESSED_DIR / "dashboard_rent_summary.parquet"
DASHBOARD_MACRO_MONTHLY_PATH = PROCESSED_DIR / "dashboard_macro_monthly.parquet"
DASHBOARD_TRADE_DETAIL_PATH = PROCESSED_DIR / "dashboard_trade_detail.parquet"
DASHBOARD_RENT_DETAIL_PATH = PROCESSED_DIR / "dashboard_rent_detail.parquet"


def optional_import(module_name: str):
    """모듈이 설치된 경우에만 import 한다."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """가중평균을 안전하게 계산한다."""
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan

    valid_values = values[mask].astype(float)
    valid_weights = weights[mask].astype(float)
    total_weight = valid_weights.sum()
    if total_weight == 0:
        return np.nan
    return float((valid_values * valid_weights).sum() / total_weight)


def _normalize_ym_text(value: str | int | None) -> str | None:
    """연월 값을 YYYYMM 문자열로 정규화한다."""
    if value is None:
        return None
    return (
        pd.Series([value], dtype="string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
        .iloc[0]
    )


def _read_parquet_optional_columns(
    path,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """필요 컬럼만 읽되 실패하면 전체 로드 후 교집합만 남긴다."""
    if columns is None:
        return pd.read_parquet(path)

    requested = list(columns)
    try:
        return pd.read_parquet(path, columns=requested)
    except Exception:
        df = pd.read_parquet(path)
        available = [column for column in requested if column in df.columns]
        if not available:
            return pd.DataFrame()
        return df[available].copy()


def _should_use_dashboard_monthly_path(start_ym: str | None, dashboard_path) -> bool:
    """요청 범위가 2020-01 이후면 대시보드 경량 파일을 사용한다."""
    normalized = _normalize_ym_text(start_ym)
    return normalized is not None and normalized >= DASHBOARD_START_YM and dashboard_path.exists()


def _should_use_dashboard_year_slice(years: Sequence[int] | None, dashboard_path) -> bool:
    """요청 연도가 전부 2020년 이후면 대시보드 상세 파일을 사용한다."""
    if not years or not dashboard_path.exists():
        return False
    return min(int(year) for year in years) >= DASHBOARD_START_YEAR


def _dashboard_detail_path(prefix: str):
    """상세 데이터 prefix에 대응하는 대시보드 파일 경로를 반환한다."""
    if prefix == "apt_trade":
        return DASHBOARD_TRADE_DETAIL_PATH
    if prefix == "apt_rent":
        return DASHBOARD_RENT_DETAIL_PATH
    return None


def _read_chunked_dataset(
    prefix: str,
    years: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """연도 조각 parquet를 읽어 하나의 DataFrame으로 합친다."""
    target_years = {int(year) for year in years} if years else None

    dashboard_path = _dashboard_detail_path(prefix)
    if dashboard_path and _should_use_dashboard_year_slice(years, dashboard_path):
        df = _read_parquet_optional_columns(dashboard_path, columns)
        if target_years and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"].dt.year.isin(target_years)].copy()
        return df

    files = sorted(PROCESSED_DIR.glob(f"{prefix}_*.parquet"))
    if target_years:
        filtered_files = []
        for file_path in files:
            try:
                file_year = int(file_path.stem.rsplit("_", maxsplit=1)[-1])
            except ValueError:
                continue
            if file_year in target_years:
                filtered_files.append(file_path)
        files = filtered_files

    if not files:
        full_path = PROCESSED_DIR / f"{prefix}.parquet"
        if not full_path.exists():
            return pd.DataFrame()
        df = _read_parquet_optional_columns(full_path, columns)
        if target_years and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"].dt.year.isin(target_years)].copy()
        return df

    frames = [_read_parquet_optional_columns(file_path, columns) for file_path in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ensure_month_columns(df: pd.DataFrame, ym_column: str = "ym") -> pd.DataFrame:
    """연월 문자열과 datetime 컬럼을 보강한다."""
    if df.empty:
        return df.copy()

    result = df.copy()
    if ym_column in result.columns:
        result[ym_column] = (
            result[ym_column]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(6)
        )
        result["date"] = pd.to_datetime(result[ym_column], format="%Y%m", errors="coerce")
    elif "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result[ym_column] = result["date"].dt.strftime("%Y%m")

    if "date" in result.columns:
        result["year"] = result["date"].dt.year
        result["month"] = result["date"].dt.month
    return result


def get_region_code(region_name: str) -> str | None:
    """지역명으로 5자리 지역 코드를 찾는다."""
    for code, name in ALL_REGIONS.items():
        if name == region_name:
            return code
    return None


def get_region_name(region_code: str) -> str:
    """지역 코드로 사용자 표시명을 반환한다."""
    return ALL_REGIONS.get(str(region_code), str(region_code))


def get_scope_codes(scope_name: str) -> list[str]:
    """서울/경기/수도권 또는 개별 지역명을 지역 코드 리스트로 변환한다."""
    if scope_name == "서울 전체":
        return list(SEOUL_REGIONS.keys())
    if scope_name == "경기 전체":
        return list(GYEONGGI_REGIONS.keys())
    if scope_name == "수도권 전체":
        return list(ALL_REGIONS.keys())

    region_code = get_region_code(scope_name)
    return [region_code] if region_code else []


def infer_scope_name(region_codes: Sequence[str] | None) -> str:
    """선택 코드 목록을 사람이 읽을 수 있는 범위명으로 바꾼다."""
    if not region_codes:
        return "수도권 전체"

    code_set = set(region_codes)
    if code_set == set(SEOUL_REGIONS.keys()):
        return "서울 전체"
    if code_set == set(GYEONGGI_REGIONS.keys()):
        return "경기 전체"
    if len(code_set) == 1:
        return get_region_name(next(iter(code_set)))
    return "선택 지역"


def _needs_region_normalization(df: pd.DataFrame) -> bool:
    """기존 잘못된 동 단위 집계 파일인지 판별한다."""
    if df.empty or "_region_name" not in df.columns or "_lawd_cd" not in df.columns:
        return False

    known_names = set(ALL_REGIONS.values())
    sample_names = set(df["_region_name"].dropna().astype(str).unique())
    return not sample_names.issubset(known_names)


def _weighted_groupby(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    weight_col: str,
    weighted_cols: Sequence[str],
) -> pd.DataFrame:
    """동일 가중치 컬럼 기준으로 여러 컬럼을 집계한다."""
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, weight_col, *weighted_cols])

    rows: list[dict[str, object]] = []
    grouped = df.groupby(list(group_cols), dropna=False, observed=True, sort=True)
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, keys)}
        row[weight_col] = float(group[weight_col].sum())
        for column in weighted_cols:
            row[column] = weighted_average(group[column], group[weight_col]) if column in group.columns else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def normalize_trade_summary(df: pd.DataFrame) -> pd.DataFrame:
    """월별 매매 집계를 자치구/시군구 단위로 정규화한다."""
    if df.empty:
        return df.copy()

    result = ensure_month_columns(df)
    result["_lawd_cd"] = result["_lawd_cd"].astype("string").str.zfill(5)
    result["_region_name"] = result["_lawd_cd"].map(ALL_REGIONS).fillna(result["_region_name"])

    duplicated_groups = result.groupby(["ym", "_lawd_cd"], observed=True).size().max() > 1
    if not _needs_region_normalization(result) and not duplicated_groups:
        return result.sort_values(["ym", "_lawd_cd"]).reset_index(drop=True)

    group_cols = ["ym", "date", "_lawd_cd", "_region_name"]
    normalized = _weighted_groupby(result, group_cols, "거래건수", TRADE_WEIGHTED_COLUMNS)
    normalized["거래건수"] = normalized["거래건수"].round().astype(int)
    return normalized.sort_values(["ym", "_lawd_cd"]).reset_index(drop=True)


def normalize_rent_summary(df: pd.DataFrame) -> pd.DataFrame:
    """월별 전월세 집계를 자치구/시군구 단위로 정규화한다."""
    if df.empty:
        return df.copy()

    result = ensure_month_columns(df)
    result["_lawd_cd"] = result["_lawd_cd"].astype("string").str.zfill(5)
    result["_region_name"] = result["_lawd_cd"].map(ALL_REGIONS).fillna(result["_region_name"])

    duplicated_groups = result.groupby(["ym", "_lawd_cd", "rentType"], observed=True).size().max() > 1
    if not _needs_region_normalization(result) and not duplicated_groups:
        return result.sort_values(["ym", "_lawd_cd", "rentType"]).reset_index(drop=True)

    group_cols = ["ym", "date", "_lawd_cd", "_region_name", "rentType"]
    normalized = _weighted_groupby(result, group_cols, "거래건수", RENT_WEIGHTED_COLUMNS)
    normalized["거래건수"] = normalized["거래건수"].round().astype(int)
    return normalized.sort_values(["ym", "_lawd_cd", "rentType"]).reset_index(drop=True)


def load_trade_summary_df(start_ym: str | None = ANALYSIS_START_YM) -> pd.DataFrame:
    """정규화된 월별 매매 집계 데이터를 로드한다."""
    path = DASHBOARD_TRADE_SUMMARY_PATH if _should_use_dashboard_monthly_path(start_ym, DASHBOARD_TRADE_SUMMARY_PATH) else PROCESSED_DIR / "monthly_trade_summary.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = normalize_trade_summary(pd.read_parquet(path))
    normalized_start = _normalize_ym_text(start_ym)
    if normalized_start:
        df = df[df["ym"] >= normalized_start].copy()
    if not df.empty:
        df["평균거래금액_전용면적당"] = df["평균거래금액"] / df["평균전용면적"].replace(0, np.nan)
    return df.sort_values(["ym", "_lawd_cd"]).reset_index(drop=True)


def load_rent_summary_df(start_ym: str | None = ANALYSIS_START_YM) -> pd.DataFrame:
    """정규화된 월별 전월세 집계 데이터를 로드한다."""
    path = DASHBOARD_RENT_SUMMARY_PATH if _should_use_dashboard_monthly_path(start_ym, DASHBOARD_RENT_SUMMARY_PATH) else PROCESSED_DIR / "monthly_rent_summary.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = normalize_rent_summary(pd.read_parquet(path))
    normalized_start = _normalize_ym_text(start_ym)
    if normalized_start:
        df = df[df["ym"] >= normalized_start].copy()
    return df.sort_values(["ym", "_lawd_cd", "rentType"]).reset_index(drop=True)


def load_macro_monthly_df(start_ym: str | None = ANALYSIS_START_YM) -> pd.DataFrame:
    """월별 거시지표 통합 테이블을 로드한다."""
    path = DASHBOARD_MACRO_MONTHLY_PATH if _should_use_dashboard_monthly_path(start_ym, DASHBOARD_MACRO_MONTHLY_PATH) else PROCESSED_DIR / "macro_monthly.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = ensure_month_columns(df, ym_column="ym")
    normalized_start = _normalize_ym_text(start_ym)
    if normalized_start:
        df = df[df["ym"] >= normalized_start].copy()
    return df.sort_values("date").reset_index(drop=True)


def load_apartment_info_df() -> pd.DataFrame:
    """건축물대장 요약 lookup table을 로드한다."""
    path = PROCESSED_DIR / "apartment_info.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_trade_detail_df(
    years: Sequence[int] | None = None,
    region_codes: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """전처리된 매매 상세 조각을 읽고 지역 컬럼을 보강한다."""
    base_columns = {"date", "dong_repr"}
    requested_columns = set(columns or [])
    requested_columns.update(base_columns)
    df = _read_chunked_dataset("apt_trade", years=years, columns=sorted(requested_columns))
    if df.empty:
        return df

    df = df.copy()
    df["region_code"] = df["dong_repr"].str.extract(r"\((\d+)\)").astype("string")
    df["region_name"] = df["region_code"].map(ALL_REGIONS).fillna(df["region_code"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ym"] = df["date"].dt.strftime("%Y%m")
    df["year"] = df["date"].dt.year
    if region_codes:
        code_set = {str(code) for code in region_codes}
        df = df[df["region_code"].isin(code_set)].copy()
    return df.sort_values("date").reset_index(drop=True)


def load_rent_detail_df(
    years: Sequence[int] | None = None,
    region_codes: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """전처리된 전월세 상세 조각을 읽고 지역 컬럼을 보강한다."""
    base_columns = {"date", "dong_repr"}
    requested_columns = set(columns or [])
    requested_columns.update(base_columns)
    df = _read_chunked_dataset("apt_rent", years=years, columns=sorted(requested_columns))
    if df.empty:
        return df

    df = df.copy()
    df["region_code"] = df["dong_repr"].str.extract(r"\((\d+)\)").astype("string")
    df["region_name"] = df["region_code"].map(ALL_REGIONS).fillna(df["region_code"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ym"] = df["date"].dt.strftime("%Y%m")
    df["year"] = df["date"].dt.year
    if region_codes:
        code_set = {str(code) for code in region_codes}
        df = df[df["region_code"].isin(code_set)].copy()
    return df.sort_values("date").reset_index(drop=True)


def aggregate_trade_scope(
    trade_df: pd.DataFrame,
    region_codes: Sequence[str] | None,
    scope_name: str | None = None,
) -> pd.DataFrame:
    """선택된 지역 코드를 하나의 범위로 묶어 월별 시계열을 만든다."""
    if trade_df.empty:
        return pd.DataFrame()

    codes = {str(code) for code in region_codes} if region_codes else set(trade_df["_lawd_cd"].astype(str))
    subset = trade_df[trade_df["_lawd_cd"].astype(str).isin(codes)].copy()
    if subset.empty:
        return pd.DataFrame()

    aggregated = _weighted_groupby(subset, ["ym", "date"], "거래건수", TRADE_WEIGHTED_COLUMNS)
    aggregated["거래건수"] = aggregated["거래건수"].round().astype(int)
    aggregated["scope_name"] = scope_name or infer_scope_name(sorted(codes))
    aggregated["평균거래금액_전용면적당"] = aggregated["평균거래금액"] / aggregated["평균전용면적"].replace(0, np.nan)
    return aggregated.sort_values("date").reset_index(drop=True)


def aggregate_rent_scope(
    rent_df: pd.DataFrame,
    region_codes: Sequence[str] | None,
    scope_name: str | None = None,
) -> pd.DataFrame:
    """선택된 지역 코드를 하나의 범위로 묶어 월별 전월세 시계열을 만든다."""
    if rent_df.empty:
        return pd.DataFrame()

    codes = {str(code) for code in region_codes} if region_codes else set(rent_df["_lawd_cd"].astype(str))
    subset = rent_df[rent_df["_lawd_cd"].astype(str).isin(codes)].copy()
    if subset.empty:
        return pd.DataFrame()

    aggregated = _weighted_groupby(subset, ["ym", "date", "rentType"], "거래건수", RENT_WEIGHTED_COLUMNS)
    aggregated["거래건수"] = aggregated["거래건수"].round().astype(int)
    aggregated["scope_name"] = scope_name or infer_scope_name(sorted(codes))
    return aggregated.sort_values(["date", "rentType"]).reset_index(drop=True)


def classify_age(age: float | int | None) -> str:
    """건물 경과연수를 연령 구간으로 분류한다."""
    if pd.isna(age):
        return "미상"
    if age <= 5:
        return AGE_LABELS[0]
    if age <= 15:
        return AGE_LABELS[1]
    if age <= 30:
        return AGE_LABELS[2]
    return AGE_LABELS[3]


def add_seoul_coordinates(df: pd.DataFrame, code_col: str = "_lawd_cd") -> pd.DataFrame:
    """서울 자치구 코드에 대한 중심 좌표를 붙인다."""
    if df.empty:
        return df.copy()

    result = df.copy()
    result["lat"] = result[code_col].astype(str).map(lambda code: SEOUL_DISTRICT_COORDS.get(code, (np.nan, np.nan))[0])
    result["lon"] = result[code_col].astype(str).map(lambda code: SEOUL_DISTRICT_COORDS.get(code, (np.nan, np.nan))[1])
    return result

