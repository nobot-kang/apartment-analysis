"""월별 집계 파이프라인.

Raw 데이터를 읽어 월별 × 지역별 요약 통계와 대시보드 전용 경량 parquet를 생성한다.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    ALL_REGIONS,
    ECOS_RAW_DIR,
    GYEONGGI_REGIONS,
    MARKET_RAW_DIR,
    MOLIT_RAW_DIR,
    PROCESSED_DIR,
    SEOUL_REGIONS,
)
from pipelines.representative_complex_pipeline import RepresentativeComplexPipeline


class AggregationPipeline:
    """Raw 데이터를 집계하여 processed 파일을 생성하는 파이프라인."""

    AREA_BINS: list[float] = [0, 60, 85, float("inf")]
    AREA_LABELS: list[str] = ["60㎡이하", "60~85㎡", "85㎡초과"]
    DASHBOARD_START_YM: str = "202001"
    DASHBOARD_START_DATE = pd.Timestamp("2020-01-01")
    TRADE_WEIGHTED_COLUMNS: list[str] = [
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
    RENT_WEIGHTED_COLUMNS: list[str] = [
        "평균보증금",
        "중앙값보증금",
        "평균월세",
        "중앙값월세",
        "평균84환산보증금",
    ]
    DASHBOARD_TRADE_DETAIL_COLUMNS: list[str] = [
        "date",
        "price",
        "area",
        "floor",
        "age",
        "apt_name_repr",
        "dong_repr",
    ]
    DASHBOARD_RENT_DETAIL_COLUMNS: list[str] = [
        "date",
        "deposit",
        "monthly_rent",
        "rentType",
        "apt_name_repr",
        "dong_repr",
    ]

    def __init__(
        self,
        molit_dir: str | Path | None = None,
        ecos_dir: str | Path | None = None,
        market_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.molit_dir = Path(molit_dir) if molit_dir else MOLIT_RAW_DIR
        self.ecos_dir = Path(ecos_dir) if ecos_dir else ECOS_RAW_DIR
        self.market_dir = Path(market_dir) if market_dir else MARKET_RAW_DIR
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _trimmed_mean(series: pd.Series, trim_pct: float = 0.1) -> float:
        """상하위 일정 비율을 절사한 평균을 계산한다."""
        sorted_values = series.dropna().sort_values()
        if len(sorted_values) == 0:
            return np.nan

        trim_count = int(len(sorted_values) * trim_pct)
        if trim_count > 0:
            sorted_values = sorted_values.iloc[trim_count:-trim_count]
        return float(sorted_values.mean()) if len(sorted_values) > 0 else np.nan

    @staticmethod
    def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
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

    def _read_parquet_with_optional_columns(
        self,
        file_path: Path,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """필요 컬럼만 읽되 스키마 차이가 있으면 전체 로드 후 교집합만 남긴다."""
        if columns is None:
            return pd.read_parquet(file_path)

        try:
            return pd.read_parquet(file_path, columns=columns)
        except Exception:
            df = pd.read_parquet(file_path)
            available_columns = [column for column in columns if column in df.columns]
            if not available_columns:
                return pd.DataFrame()
            return df[available_columns].copy()

    def _load_processed_chunks(
        self,
        prefix: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """prefix로 시작하는 모든 전처리 parquet 조각을 읽어 합친다."""
        files = sorted(self.output_dir.glob(f"{prefix}_*.parquet"))
        if not files:
            full_path = self.output_dir / f"{prefix}.parquet"
            if full_path.exists():
                return self._read_parquet_with_optional_columns(full_path, columns)
            logger.warning(f"전처리된 {prefix} 데이터 조각을 찾을 수 없습니다.")
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for file_path in files:
            try:
                frames.append(self._read_parquet_with_optional_columns(file_path, columns))
            except Exception as exc:
                logger.error(f"조각 로드 실패: {file_path.name} - {exc}")

        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _filter_dashboard_window(
        self,
        df: pd.DataFrame,
        ym_col: str | None = None,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """대시보드에서 사용하는 2020-01 이후 데이터만 남긴다."""
        if df.empty:
            return df.copy()

        result = df.copy()
        if ym_col and ym_col in result.columns:
            result[ym_col] = (
                result[ym_col]
                .astype("string")
                .str.replace(r"\.0$", "", regex=True)
                .str.replace(r"\D", "", regex=True)
                .str.zfill(6)
            )
            return result[result[ym_col] >= self.DASHBOARD_START_YM].copy()

        if date_col in result.columns:
            result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
            return result[result[date_col] >= self.DASHBOARD_START_DATE].copy()

        return result

    def _read_output_parquet(self, name: str) -> pd.DataFrame:
        """output_dir 아래 parquet를 읽는다."""
        path = self.output_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _write_output_parquet(self, name: str, df: pd.DataFrame) -> None:
        """output_dir 아래 parquet를 저장한다."""
        if df.empty and len(df.columns) == 0:
            logger.warning(f"빈 데이터셋이라 저장하지 않음: {name}")
            return
        df.to_parquet(self.output_dir / name, index=False)

    def _weighted_scope_groupby(
        self,
        df: pd.DataFrame,
        group_cols: Sequence[str],
        weight_col: str,
        weighted_cols: Sequence[str],
    ) -> pd.DataFrame:
        """여러 지역을 하나의 scope로 묶어 월별 가중 집계를 계산한다."""
        if df.empty:
            return pd.DataFrame(columns=[*group_cols, weight_col, *weighted_cols])

        rows: list[dict[str, object]] = []
        for key, group in df.groupby(list(group_cols), observed=True, dropna=False, sort=True):
            keys = key if isinstance(key, tuple) else (key,)
            row = {column: value for column, value in zip(group_cols, keys)}
            row[weight_col] = int(group[weight_col].sum())
            for column in weighted_cols:
                row[column] = self._weighted_average(group[column], group[weight_col]) if column in group.columns else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _get_scope_definitions(self) -> list[tuple[str, list[str]]]:
        """대시보드에서 지원하는 scope 정의를 반환한다."""
        scopes = [
            ("서울 전체", list(SEOUL_REGIONS.keys())),
            ("경기 전체", list(GYEONGGI_REGIONS.keys())),
            ("수도권 전체", list(ALL_REGIONS.keys())),
        ]
        scopes.extend((name, [code]) for code, name in sorted(ALL_REGIONS.items(), key=lambda item: item[1]))
        return scopes

    def _aggregate_trade_scope(
        self,
        trade_df: pd.DataFrame,
        region_codes: Sequence[str],
        scope_name: str,
    ) -> pd.DataFrame:
        """월별 매매 집계를 scope 단위로 묶는다."""
        codes = {str(code) for code in region_codes}
        subset = trade_df[trade_df["_lawd_cd"].astype(str).isin(codes)].copy()
        if subset.empty:
            return pd.DataFrame()

        aggregated = self._weighted_scope_groupby(
            subset,
            ["ym", "date"],
            "거래건수",
            self.TRADE_WEIGHTED_COLUMNS,
        )
        aggregated["scope_name"] = scope_name
        aggregated["평균거래금액_전용면적당"] = aggregated["평균거래금액"] / aggregated["평균전용면적"].replace(0, np.nan)
        return aggregated.sort_values("date").reset_index(drop=True)

    def _aggregate_rent_scope(
        self,
        rent_df: pd.DataFrame,
        region_codes: Sequence[str],
        scope_name: str,
    ) -> pd.DataFrame:
        """월별 전월세 집계를 scope 단위로 묶는다."""
        codes = {str(code) for code in region_codes}
        subset = rent_df[rent_df["_lawd_cd"].astype(str).isin(codes)].copy()
        if subset.empty:
            return pd.DataFrame()

        aggregated = self._weighted_scope_groupby(
            subset,
            ["ym", "date", "rentType"],
            "거래건수",
            self.RENT_WEIGHTED_COLUMNS,
        )
        aggregated["scope_name"] = scope_name
        return aggregated.sort_values(["date", "rentType"]).reset_index(drop=True)

    def build_monthly_trade_summary(self) -> pd.DataFrame:
        """전처리된 매매 데이터를 로드하여 월별 × 지역별 집계를 생성한다."""
        logger.info("매매 월별 집계 시작 (전처리 데이터 조각 사용)")
        df = self._load_processed_chunks(
            "apt_trade",
            columns=["date", "price", "price_std84", "area", "age", "dong_repr"],
        )
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "price", "area", "dong_repr"])
        df["ym"] = df["date"].dt.strftime("%Y%m")
        df["_lawd_cd"] = df["dong_repr"].str.extract(r"\((\d+)\)")
        df["_region_name"] = df["_lawd_cd"].map(ALL_REGIONS).fillna(df["_lawd_cd"])

        q_low = df["price"].quantile(0.01)
        q_high = df["price"].quantile(0.99)
        df = df[df["price"].between(q_low, q_high)].copy()
        df["area_group"] = pd.cut(df["area"], bins=self.AREA_BINS, labels=self.AREA_LABELS, right=True)

        group_cols = ["ym", "_lawd_cd", "_region_name"]
        summary = (
            df.groupby(group_cols, observed=True)
            .agg(
                거래건수=("price", "size"),
                평균거래금액=("price", "mean"),
                중앙값거래금액=("price", "median"),
                평균84환산금액=("price_std84", "mean"),
                중앙값84환산금액=("price_std84", "median"),
            )
            .reset_index()
        )
        summary["date"] = pd.to_datetime(summary["ym"], format="%Y%m", errors="coerce")

        trim_means = (
            df.groupby(group_cols, observed=True)["price"]
            .apply(self._trimmed_mean)
            .reset_index(name="절사평균거래금액")
        )
        summary = summary.merge(trim_means, on=group_cols, how="left")

        avg_area = df.groupby(group_cols, observed=True)["area"].mean().reset_index(name="평균전용면적")
        summary = summary.merge(avg_area, on=group_cols, how="left")

        avg_age = df.groupby(group_cols, observed=True)["age"].mean().reset_index(name="평균건물연령")
        summary = summary.merge(avg_age, on=group_cols, how="left")

        area_avg = (
            df.groupby(group_cols + ["area_group"], observed=True)["price"]
            .mean()
            .reset_index()
            .pivot_table(index=group_cols, columns="area_group", values="price", observed=True)
            .reset_index()
        )
        area_avg.columns = [
            f"평균거래금액_{column}" if column in self.AREA_LABELS else column
            for column in area_avg.columns
        ]
        summary = summary.merge(area_avg, on=group_cols, how="left")
        summary = summary[[
            "ym",
            "date",
            "_lawd_cd",
            "_region_name",
            "거래건수",
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
        ]]

        self._write_output_parquet("monthly_trade_summary.parquet", summary)
        logger.info(f"매매 집계 저장 완료: monthly_trade_summary.parquet ({len(summary)}건)")
        return summary

    def build_monthly_rent_summary(self) -> pd.DataFrame:
        """전처리된 전월세 데이터를 로드하여 월별 × 지역별 집계를 생성한다."""
        logger.info("전월세 월별 집계 시작 (전처리 데이터 조각 사용)")
        df = self._load_processed_chunks(
            "apt_rent",
            columns=["date", "deposit", "monthly_rent", "deposit_std84", "dong_repr", "rentType"],
        )
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "deposit", "dong_repr"])
        if "rentType" not in df.columns:
            df["rentType"] = np.where(df["monthly_rent"].fillna(0) == 0, "전세", "월세")

        df["ym"] = df["date"].dt.strftime("%Y%m")
        df["_lawd_cd"] = df["dong_repr"].str.extract(r"\((\d+)\)")
        df["_region_name"] = df["_lawd_cd"].map(ALL_REGIONS).fillna(df["_lawd_cd"])

        group_cols = ["ym", "_lawd_cd", "_region_name", "rentType"]
        summary = (
            df.groupby(group_cols, observed=True)
            .agg(
                거래건수=("deposit", "size"),
                평균보증금=("deposit", "mean"),
                중앙값보증금=("deposit", "median"),
                평균월세=("monthly_rent", "mean"),
                중앙값월세=("monthly_rent", "median"),
                평균84환산보증금=("deposit_std84", "mean"),
            )
            .reset_index()
        )
        summary["date"] = pd.to_datetime(summary["ym"], format="%Y%m", errors="coerce")
        summary = summary[[
            "ym",
            "date",
            "_lawd_cd",
            "_region_name",
            "rentType",
            "거래건수",
            "평균보증금",
            "중앙값보증금",
            "평균월세",
            "중앙값월세",
            "평균84환산보증금",
        ]]

        self._write_output_parquet("monthly_rent_summary.parquet", summary)
        logger.info(f"전월세 집계 저장 완료: monthly_rent_summary.parquet ({len(summary)}건)")
        return summary

    def build_macro_monthly(self) -> pd.DataFrame:
        """ECOS + yfinance raw 데이터를 월별 통합 테이블로 병합한다."""
        logger.info("거시지표 월별 통합 시작")
        merged: pd.DataFrame | None = None

        for file_path in sorted(self.ecos_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(file_path)
                label = file_path.stem
                df = df[["date", "value"]].rename(columns={"value": label})
                merged = df if merged is None else merged.merge(df, on="date", how="outer")
            except Exception as exc:
                logger.warning(f"ECOS 로드 실패: {file_path.name} - {exc}")

        for file_path in sorted(self.market_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(file_path)
                label = file_path.stem
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
                df = df[["date", "close"]].rename(columns={"close": label})
                merged = df if merged is None else merged.merge(df, on="date", how="outer")
            except Exception as exc:
                logger.warning(f"yfinance 로드 실패: {file_path.name} - {exc}")

        if merged is None or merged.empty:
            logger.warning("거시지표 데이터가 없어 집계를 건너뜁니다.")
            return pd.DataFrame()

        merged = merged.sort_values("date").reset_index(drop=True)
        merged["ym"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y%m")
        self._write_output_parquet("macro_monthly.parquet", merged)
        logger.info(f"거시지표 통합 저장 완료: macro_monthly.parquet ({len(merged)}건)")
        return merged

    def build_dashboard_jeonse_ratio(self, trade_summary: pd.DataFrame, rent_summary: pd.DataFrame) -> pd.DataFrame:
        """전세가율 선계산 파일을 생성한다."""
        if trade_summary.empty or rent_summary.empty:
            return pd.DataFrame()

        trade = self._filter_dashboard_window(trade_summary, ym_col="ym")
        rent = self._filter_dashboard_window(rent_summary, ym_col="ym")
        jeonse = rent[rent["rentType"] == "전세"][["ym", "date", "_lawd_cd", "_region_name", "평균보증금", "거래건수"]].copy()
        merged = trade.merge(
            jeonse,
            on=["ym", "date", "_lawd_cd", "_region_name"],
            how="inner",
            suffixes=("_trade", "_jeonse"),
        )
        merged["전세가율"] = (merged["평균보증금"] / merged["평균거래금액"]) * 100
        self._write_output_parquet("dashboard_jeonse_ratio_monthly.parquet", merged)
        return merged.sort_values(["_lawd_cd", "ym"]).reset_index(drop=True)

    def build_dashboard_conversion_rate(self, rent_summary: pd.DataFrame) -> pd.DataFrame:
        """scope 단위 전월세 전환율 선계산 파일을 생성한다."""
        if rent_summary.empty:
            return pd.DataFrame()

        rent = self._filter_dashboard_window(rent_summary, ym_col="ym")
        frames: list[pd.DataFrame] = []
        for scope_name, codes in self._get_scope_definitions():
            rent_scope = self._aggregate_rent_scope(rent, codes, scope_name)
            if rent_scope.empty:
                continue
            jeonse = rent_scope[rent_scope["rentType"] == "전세"][["ym", "date", "평균보증금", "scope_name"]].rename(columns={"평균보증금": "jeonse_deposit"})
            wolse = rent_scope[rent_scope["rentType"] == "월세"][["ym", "date", "평균보증금", "평균월세", "거래건수", "scope_name"]].rename(columns={"평균보증금": "wolse_deposit", "거래건수": "sample_count"})
            merged = wolse.merge(jeonse, on=["ym", "date", "scope_name"], how="left")
            merged["deposit_gap"] = merged["jeonse_deposit"] - merged["wolse_deposit"]
            merged = merged[merged["deposit_gap"] > 0].copy()
            merged["conversion_rate"] = (merged["평균월세"] * 12 / merged["deposit_gap"]) * 100
            merged = merged[(merged["conversion_rate"] >= 0) & (merged["conversion_rate"] <= 30)]
            frames.append(merged)

        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._write_output_parquet("dashboard_conversion_rate_monthly.parquet", result)
        return result.sort_values(["scope_name", "date"]).reset_index(drop=True) if not result.empty else result

    def build_dashboard_district_year_metrics(self, trade_summary: pd.DataFrame) -> pd.DataFrame:
        """연도별 지역 랭킹/히트맵용 선계산 파일을 생성한다."""
        if trade_summary.empty:
            return pd.DataFrame()

        trade = self._filter_dashboard_window(trade_summary, ym_col="ym").copy()
        if trade.empty:
            return pd.DataFrame()

        if "date" in trade.columns:
            trade["date"] = pd.to_datetime(trade["date"], errors="coerce")
            trade["year"] = trade["date"].dt.year
        else:
            trade["year"] = trade["ym"].astype(str).str[:4].astype(int)

        if "평균거래금액_전용면적당" not in trade.columns:
            trade["평균거래금액_전용면적당"] = trade["평균거래금액"] / trade["평균전용면적"].replace(0, np.nan)

        working = trade[["year", "_lawd_cd", "_region_name", "평균거래금액", "평균거래금액_전용면적당", "거래건수"]].copy()
        working["weighted_price"] = working["평균거래금액"] * working["거래건수"]
        working["weighted_price_per_m2"] = working["평균거래금액_전용면적당"] * working["거래건수"]
        yearly = (
            working.groupby(["year", "_lawd_cd", "_region_name"], observed=True)
            .agg(
                weighted_price=("weighted_price", "sum"),
                weighted_price_per_m2=("weighted_price_per_m2", "sum"),
                trade_count=("거래건수", "sum"),
            )
            .reset_index()
        )
        yearly = yearly[yearly["trade_count"] > 0].copy()
        yearly["avg_price"] = yearly["weighted_price"] / yearly["trade_count"]
        yearly["avg_price_per_m2"] = yearly["weighted_price_per_m2"] / yearly["trade_count"]
        self._write_output_parquet("dashboard_district_year_metrics.parquet", yearly)
        return yearly

    def build_dashboard_cycle_features(self, trade_summary: pd.DataFrame, macro_monthly: pd.DataFrame) -> pd.DataFrame:
        """서울 전체 시장 사이클 특징량 선계산 파일을 생성한다."""
        if trade_summary.empty or macro_monthly.empty:
            return pd.DataFrame()

        trade = self._filter_dashboard_window(trade_summary, ym_col="ym")
        macro = self._filter_dashboard_window(macro_monthly, ym_col="ym")
        seoul_trade = self._aggregate_trade_scope(trade, list(SEOUL_REGIONS.keys()), "서울 전체")
        if seoul_trade.empty:
            return pd.DataFrame()

        combined = seoul_trade.merge(macro, on=["ym", "date"], how="inner").sort_values("date").reset_index(drop=True)
        combined["rate_direction"] = np.sign(combined["bok_rate"].diff()) if "bok_rate" in combined.columns else 0
        combined["m2_yoy"] = combined["m2"].pct_change(12) * 100 if "m2" in combined.columns else np.nan
        combined["vol_mom"] = combined["거래건수"].pct_change() * 100
        combined["vol_mom_3ma"] = combined["vol_mom"].rolling(3).mean()
        combined["price_yoy"] = combined["평균거래금액"].pct_change(12) * 100

        def classify(row: pd.Series) -> str:
            if row.get("rate_direction", 0) <= 0 and row.get("m2_yoy", 0) >= 5 and row.get("vol_mom_3ma", 0) >= 0:
                return "과열"
            if row.get("rate_direction", 0) >= 1 and row.get("m2_yoy", 0) <= 2 and row.get("vol_mom_3ma", 0) <= -5:
                return "침체"
            if row.get("rate_direction", 0) >= 1 and row.get("vol_mom_3ma", 0) <= 0:
                return "조정"
            return "회복"

        combined["phase_rule"] = combined.apply(classify, axis=1)
        result = combined.dropna(subset=["price_yoy", "m2_yoy", "vol_mom_3ma"]).reset_index(drop=True)
        self._write_output_parquet("dashboard_cycle_features.parquet", result)
        return result

    def build_dashboard_trade_anomalies(self) -> pd.DataFrame:
        """대시보드용 이상거래 선계산 파일을 생성한다."""
        trade_detail = self._read_output_parquet("dashboard_trade_detail.parquet")
        if trade_detail.empty:
            return pd.DataFrame()

        df = trade_detail.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["region_code"] = df["dong_repr"].str.extract(r"\((\d+)\)").astype("string")
        df["region_name"] = df["region_code"].map(ALL_REGIONS).fillna(df["region_code"])
        df["ym"] = df["date"].dt.strftime("%Y%m")
        df["year"] = df["date"].dt.year
        df["price_per_m2"] = df["price"] / df["area"].replace(0, np.nan)
        df = df.dropna(subset=["price_per_m2", "area", "floor"])
        if df.empty:
            return df

        sklearn_ensemble = None
        try:
            sklearn_ensemble = importlib.import_module("sklearn.ensemble")
        except Exception:
            sklearn_ensemble = None

        result_frames: list[pd.DataFrame] = []
        group_cols = ["region_code", "year"]
        for _, group in df.groupby(group_cols, observed=True):
            working = group.copy()
            features = working[["price_per_m2", "area", "floor", "age"]].fillna(working[["price_per_m2", "area", "floor", "age"]].median())
            if sklearn_ensemble is not None and len(working) >= 50:
                model = sklearn_ensemble.IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
                model.fit(features)
                working["anomaly_score"] = model.score_samples(features)
                working["is_anomaly"] = model.predict(features) == -1
            else:
                medians = features.median()
                mads = (features - medians).abs().median().replace(0, np.nan)
                robust_z = ((features - medians).abs() / mads).fillna(0)
                working["anomaly_score"] = robust_z.mean(axis=1)
                threshold = working["anomaly_score"].quantile(0.97)
                working["is_anomaly"] = working["anomaly_score"] >= threshold
            result_frames.append(working)

        result = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
        self._write_output_parquet("dashboard_trade_anomalies.parquet", result)
        return result

    def build_dashboard_datasets(
        self,
        trade_summary: pd.DataFrame | None = None,
        rent_summary: pd.DataFrame | None = None,
        macro_monthly: pd.DataFrame | None = None,
    ) -> None:
        """대시보드 전용 경량 parquet와 선계산 산출물을 생성한다."""
        logger.info("대시보드 전용 데이터셋 생성 시작")

        trade_summary = trade_summary if trade_summary is not None else self._read_output_parquet("monthly_trade_summary.parquet")
        rent_summary = rent_summary if rent_summary is not None else self._read_output_parquet("monthly_rent_summary.parquet")
        macro_monthly = macro_monthly if macro_monthly is not None else self._read_output_parquet("macro_monthly.parquet")

        dashboard_trade_summary = self._filter_dashboard_window(trade_summary, ym_col="ym")
        dashboard_rent_summary = self._filter_dashboard_window(rent_summary, ym_col="ym")
        dashboard_macro = self._filter_dashboard_window(macro_monthly, ym_col="ym")
        self._write_output_parquet("dashboard_trade_summary.parquet", dashboard_trade_summary)
        self._write_output_parquet("dashboard_rent_summary.parquet", dashboard_rent_summary)
        self._write_output_parquet("dashboard_macro_monthly.parquet", dashboard_macro)

        trade_detail = self._load_processed_chunks("apt_trade", columns=self.DASHBOARD_TRADE_DETAIL_COLUMNS)
        trade_detail = self._filter_dashboard_window(trade_detail, date_col="date")
        self._write_output_parquet("dashboard_trade_detail.parquet", trade_detail)

        rent_detail = self._load_processed_chunks("apt_rent", columns=self.DASHBOARD_RENT_DETAIL_COLUMNS)
        if not rent_detail.empty and "rentType" not in rent_detail.columns:
            rent_detail["rentType"] = np.where(rent_detail["monthly_rent"].fillna(0) == 0, "전세", "월세")
        rent_detail = self._filter_dashboard_window(rent_detail, date_col="date")
        self._write_output_parquet("dashboard_rent_detail.parquet", rent_detail)

        jeonse_ratio = self.build_dashboard_jeonse_ratio(dashboard_trade_summary, dashboard_rent_summary)
        conversion_rate = self.build_dashboard_conversion_rate(dashboard_rent_summary)
        district_year_metrics = self.build_dashboard_district_year_metrics(dashboard_trade_summary)
        cycle_features = self.build_dashboard_cycle_features(dashboard_trade_summary, dashboard_macro)
        trade_anomalies = self.build_dashboard_trade_anomalies()

        logger.info(
            "대시보드 데이터셋 저장 완료: "
            f"trade_summary={len(dashboard_trade_summary)}, "
            f"rent_summary={len(dashboard_rent_summary)}, "
            f"macro={len(dashboard_macro)}, "
            f"trade_detail={len(trade_detail)}, "
            f"rent_detail={len(rent_detail)}, "
            f"jeonse_ratio={len(jeonse_ratio)}, "
            f"conversion_rate={len(conversion_rate)}, "
            f"district_year_metrics={len(district_year_metrics)}, "
            f"cycle_features={len(cycle_features)}, "
            f"trade_anomalies={len(trade_anomalies)}"
        )

    def _load_apartment_list(self) -> pd.DataFrame:
        """단지 목록 원본을 로드한다."""
        path = self.molit_dir / "apartment_list.parquet"
        if not path.exists():
            logger.warning("apartment_list.parquet 가 없어 단지 마스터 구성을 건너뜁니다.")
            return pd.DataFrame()
        return pd.read_parquet(path)

    def build_complex_master(self) -> pd.DataFrame:
        """단지 정적 특성 마스터 테이블을 생성한다."""
        apt_list = self._load_apartment_list()
        apt_info = self._read_output_parquet("apartment_info.parquet")
        if apt_list.empty and apt_info.empty:
            return pd.DataFrame()

        if not apt_list.empty:
            base = (
                apt_list.sort_values("aptSeq")
                .drop_duplicates(subset=["aptSeq"], keep="first")
                [["aptSeq", "aptNm", "sggCd", "umdCd", "umdNm", "buildYear", "roadNm"]]
                .rename(
                    columns={
                        "aptNm": "apt_name_source",
                        "sggCd": "sigungu_code_source",
                        "umdCd": "bjdong_code_source",
                        "umdNm": "dong_name_source",
                        "buildYear": "build_year_source",
                        "roadNm": "road_name_source",
                    }
                )
            )
        else:
            base = pd.DataFrame(columns=["aptSeq"])

        master = base.merge(apt_info, on="aptSeq", how="outer")

        for column in [
            "land_area",
            "floor_area_ratio_total_area",
            "total_area",
            "floor_area_ratio",
            "building_coverage_ratio",
            "household_count",
            "total_parking_count",
            "ground_floor_count",
            "underground_floor_count",
            "parking_per_household",
            "avg_land_area_per_household",
            "avg_total_area_per_household",
            "redevelopment_option_score",
        ]:
            if column not in master.columns:
                master[column] = np.nan

        sigungu_series = master["sigungu_code"] if "sigungu_code" in master.columns else pd.Series(pd.NA, index=master.index)
        bjdong_series = master["bjdong_code"] if "bjdong_code" in master.columns else pd.Series(pd.NA, index=master.index)
        sigungu_source = (
            master["sigungu_code_source"] if "sigungu_code_source" in master.columns else pd.Series(pd.NA, index=master.index)
        )
        bjdong_source = (
            master["bjdong_code_source"] if "bjdong_code_source" in master.columns else pd.Series(pd.NA, index=master.index)
        )
        dong_name_source = master["dong_name_source"] if "dong_name_source" in master.columns else pd.Series(pd.NA, index=master.index)
        master["sigungu_code"] = sigungu_series.combine_first(sigungu_source)
        master["bjdong_code"] = bjdong_series.combine_first(bjdong_source)
        master["dong_name"] = master["dong_name"] if "dong_name" in master.columns else dong_name_source
        master["dong_name"] = master["dong_name"].combine_first(dong_name_source)
        apt_name_ledger = master["apt_name_ledger"] if "apt_name_ledger" in master.columns else pd.Series(pd.NA, index=master.index)
        apt_name_source = master["apt_name_source"] if "apt_name_source" in master.columns else pd.Series(pd.NA, index=master.index)
        master["apt_name"] = apt_name_ledger.combine_first(apt_name_source)
        master["build_year_source"] = pd.to_numeric(master.get("build_year_source"), errors="coerce")

        if "completion_date" in master.columns:
            master["completion_date"] = pd.to_datetime(master["completion_date"], errors="coerce")
            master["completion_year"] = master["completion_date"].dt.year
        else:
            master["completion_date"] = pd.NaT
            master["completion_year"] = np.nan
        master["completion_year"] = master["completion_year"].fillna(master["build_year_source"])

        safe_households = master["household_count"].replace(0, np.nan)
        if "total_parking_count" in master.columns:
            master["parking_per_household"] = master["total_parking_count"] / safe_households
        master["avg_land_area_per_household"] = master["avg_land_area_per_household"].where(
            master["avg_land_area_per_household"].gt(0),
            master["land_area"] / safe_households,
        )
        master["avg_total_area_per_household"] = master["avg_total_area_per_household"].where(
            master["avg_total_area_per_household"].gt(0),
            master["total_area"] / safe_households,
        )

        road_name = master.get("road_name_source", pd.Series("", index=master.index)).fillna("").astype(str).str.strip()
        dong_name = master.get("dong_name", pd.Series("", index=master.index)).fillna("").astype(str).str.strip()
        apt_name = master.get("apt_name", pd.Series("", index=master.index)).fillna("").astype(str).str.strip()
        sigungu_code = master.get("sigungu_code", pd.Series("", index=master.index)).fillna("").astype(str).str.strip()
        master["dong_repr"] = np.where(dong_name.eq(""), np.nan, dong_name + "(" + sigungu_code + ")")
        master["apt_name_repr"] = np.where(
            apt_name.eq(""),
            np.nan,
            np.where(road_name.eq(""), apt_name + "(" + dong_name + ")", apt_name + "-" + road_name + "(" + dong_name + ")"),
        )

        master["complex_scale_bucket"] = pd.cut(
            master["household_count"],
            bins=[0, 300, 1000, 2000, np.inf],
            labels=["소형", "중형", "대단지", "초대단지"],
            right=False,
        ).astype("string")
        master["density_bucket"] = pd.cut(
            master["floor_area_ratio"],
            bins=[0, 200, 300, 400, np.inf],
            labels=["저밀도", "중밀도", "고밀도", "초고밀도"],
            right=False,
        ).astype("string")

        for column in [
            "floor_area_ratio",
            "building_coverage_ratio",
            "avg_land_area_per_household",
            "avg_total_area_per_household",
            "total_parking_count",
            "parking_per_household",
        ]:
            master[f"{column}_missing"] = (~master[column].gt(0)).astype(int)

        current_year = pd.Timestamp.today().year
        complex_age = current_year - master["completion_year"]
        complex_age = complex_age.where(complex_age >= 0)
        far_input = master["floor_area_ratio"].where(master["floor_area_ratio"].gt(0))
        bcr_input = master["building_coverage_ratio"].where(master["building_coverage_ratio"].gt(0))
        land_input = master["avg_land_area_per_household"].where(master["avg_land_area_per_household"].gt(0))
        age_component = ((complex_age - 15).clip(lower=0, upper=25) / 25.0).fillna(0) * 40.0
        far_component = ((250 - far_input).clip(lower=0, upper=250) / 250.0).fillna(0) * 25.0
        bcr_component = ((35 - bcr_input).clip(lower=0, upper=35) / 35.0).fillna(0) * 15.0
        land_component = ((land_input - 20).clip(lower=0, upper=40) / 40.0).fillna(0) * 20.0
        master["redevelopment_option_score"] = (age_component + far_component + bcr_component + land_component).round(2)
        master["feature_missing_count"] = master[
            [
                "floor_area_ratio_missing",
                "building_coverage_ratio_missing",
                "avg_land_area_per_household_missing",
                "avg_total_area_per_household_missing",
                "total_parking_count_missing",
                "parking_per_household_missing",
            ]
        ].sum(axis=1)

        output_cols = [
            "aptSeq",
            "apt_name",
            "apt_name_repr",
            "dong_name",
            "dong_repr",
            "address",
            "sigungu_code",
            "bjdong_code",
            "completion_date",
            "completion_year",
            "land_area",
            "floor_area_ratio_total_area",
            "total_area",
            "floor_area_ratio",
            "building_coverage_ratio",
            "household_count",
            "total_parking_count",
            "parking_per_household",
            "avg_land_area_per_household",
            "avg_total_area_per_household",
            "ground_floor_count",
            "underground_floor_count",
            "household_count_source",
            "floor_area_ratio_source",
            "building_coverage_ratio_source",
            "total_parking_count_source",
            "parking_value_source",
            "complex_scale_bucket",
            "density_bucket",
            "redevelopment_option_score",
            "floor_area_ratio_missing",
            "building_coverage_ratio_missing",
            "avg_land_area_per_household_missing",
            "avg_total_area_per_household_missing",
            "total_parking_count_missing",
            "parking_per_household_missing",
            "feature_missing_count",
        ]
        output_cols = [column for column in output_cols if column in master.columns]
        master = master[output_cols].sort_values("aptSeq").reset_index(drop=True)
        self._write_output_parquet("complex_master.parquet", master)
        return master

    def build_complex_monthly_panel(
        self,
        complex_master: pd.DataFrame | None = None,
        macro_monthly: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """단지-월 결과/설명 패널을 생성한다."""
        complex_master = complex_master if complex_master is not None else self._read_output_parquet("complex_master.parquet")
        macro_monthly = macro_monthly if macro_monthly is not None else self._read_output_parquet("macro_monthly.parquet")
        macro_monthly = self._filter_dashboard_window(macro_monthly, ym_col="ym")

        trade = self._load_processed_chunks(
            "apt_trade",
            columns=["date", "aptSeq", "price", "price_std84", "area", "apt_name_repr", "dong_repr"],
        )
        trade = self._filter_dashboard_window(trade, date_col="date")
        if not trade.empty:
            trade = trade.copy()
            trade["date"] = pd.to_datetime(trade["date"], errors="coerce")
            trade["ym"] = trade["date"].dt.strftime("%Y%m")
            trade["price_per_m2"] = trade["price"] / trade["area"].replace(0, np.nan)
            trade_monthly = (
                trade.groupby(["aptSeq", "ym"], observed=True)
                .agg(
                    trade_count=("price", "size"),
                    trade_price_mean=("price", "mean"),
                    trade_price_std84=("price_std84", "mean"),
                    trade_price_per_m2=("price_per_m2", "mean"),
                )
                .reset_index()
            )
            trade_monthly["date"] = pd.to_datetime(trade_monthly["ym"], format="%Y%m", errors="coerce")
        else:
            trade_monthly = pd.DataFrame()

        rent = self._load_processed_chunks(
            "apt_rent",
            columns=["date", "aptSeq", "deposit", "deposit_std84", "monthly_rent", "area", "rentType"],
        )
        rent = self._filter_dashboard_window(rent, date_col="date")
        if not rent.empty:
            rent = rent.copy()
            rent["date"] = pd.to_datetime(rent["date"], errors="coerce")
            rent["ym"] = rent["date"].dt.strftime("%Y%m")
            if "rentType" not in rent.columns:
                rent["rentType"] = np.where(rent["monthly_rent"].fillna(0) == 0, "전세", "월세")
            rent["deposit_per_m2"] = rent["deposit"] / rent["area"].replace(0, np.nan)
            rent["monthly_rent_per_m2"] = rent["monthly_rent"] / rent["area"].replace(0, np.nan)

            jeonse = rent[rent["rentType"] == "전세"].copy()
            wolse = rent[rent["rentType"] == "월세"].copy()

            jeonse_monthly = (
                jeonse.groupby(["aptSeq", "ym"], observed=True)
                .agg(
                    jeonse_count=("deposit", "size"),
                    jeonse_deposit_mean=("deposit", "mean"),
                    jeonse_deposit_std84=("deposit_std84", "mean"),
                    jeonse_deposit_per_m2=("deposit_per_m2", "mean"),
                )
                .reset_index()
            ) if not jeonse.empty else pd.DataFrame()
            if not jeonse_monthly.empty:
                jeonse_monthly["date"] = pd.to_datetime(jeonse_monthly["ym"], format="%Y%m", errors="coerce")

            wolse_monthly = (
                wolse.groupby(["aptSeq", "ym"], observed=True)
                .agg(
                    wolse_count=("deposit", "size"),
                    wolse_deposit_mean=("deposit", "mean"),
                    wolse_monthly_rent_mean=("monthly_rent", "mean"),
                    wolse_monthly_rent_per_m2=("monthly_rent_per_m2", "mean"),
                )
                .reset_index()
            ) if not wolse.empty else pd.DataFrame()
            if not wolse_monthly.empty:
                wolse_monthly["date"] = pd.to_datetime(wolse_monthly["ym"], format="%Y%m", errors="coerce")
        else:
            jeonse_monthly = pd.DataFrame()
            wolse_monthly = pd.DataFrame()

        panel = trade_monthly.copy()
        for monthly_df in [jeonse_monthly, wolse_monthly]:
            if monthly_df.empty:
                continue
            if panel.empty:
                panel = monthly_df.copy()
            else:
                panel = panel.merge(monthly_df, on=["aptSeq", "ym", "date"], how="outer")

        if panel.empty:
            return panel

        if not complex_master.empty:
            panel = panel.merge(complex_master, on="aptSeq", how="left")

        for column in [
            "trade_count",
            "trade_price_mean",
            "trade_price_std84",
            "trade_price_per_m2",
            "jeonse_count",
            "jeonse_deposit_mean",
            "jeonse_deposit_std84",
            "jeonse_deposit_per_m2",
            "wolse_count",
            "wolse_deposit_mean",
            "wolse_monthly_rent_mean",
            "wolse_monthly_rent_per_m2",
            "household_count",
            "avg_land_area_per_household",
            "completion_year",
        ]:
            if column not in panel.columns:
                panel[column] = np.nan

        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel["year"] = panel["date"].dt.year
        panel["month"] = panel["date"].dt.month
        panel["completion_year"] = pd.to_numeric(panel["completion_year"], errors="coerce")
        panel["complex_age"] = panel["year"] - panel["completion_year"]
        panel.loc[panel["complex_age"] < 0, "complex_age"] = np.nan
        panel["jeonse_ratio"] = (panel["jeonse_deposit_std84"] / panel["trade_price_std84"]) * 100
        panel["deposit_gap"] = panel["jeonse_deposit_mean"] - panel["wolse_deposit_mean"]
        panel["conversion_rate"] = (panel["wolse_monthly_rent_mean"] * 12 / panel["deposit_gap"]) * 100
        panel.loc[panel["deposit_gap"] <= 0, "conversion_rate"] = np.nan
        trade_count = panel["trade_count"] if "trade_count" in panel.columns else pd.Series(0, index=panel.index, dtype="float64")
        household_count = panel["household_count"] if "household_count" in panel.columns else pd.Series(np.nan, index=panel.index, dtype="float64")
        land_share = panel["avg_land_area_per_household"] if "avg_land_area_per_household" in panel.columns else pd.Series(np.nan, index=panel.index, dtype="float64")
        price_std84 = panel["trade_price_std84"] if "trade_price_std84" in panel.columns else pd.Series(np.nan, index=panel.index, dtype="float64")
        panel["trade_occurrence"] = trade_count.fillna(0).gt(0).astype(int)
        panel["turnover_rate"] = trade_count / household_count.replace(0, np.nan)
        panel["land_value_proxy_per_py"] = price_std84 / (land_share / 3.3058)

        for column in [
            "trade_price_per_m2",
            "trade_price_std84",
            "jeonse_deposit_per_m2",
            "jeonse_deposit_std84",
            "wolse_monthly_rent_per_m2",
            "jeonse_ratio",
            "conversion_rate",
        ]:
            if column in panel.columns:
                panel[f"{column}_yoy"] = panel.groupby("aptSeq", observed=True)[column].pct_change(12, fill_method=None) * 100

        if not macro_monthly.empty:
            macro_features = macro_monthly.copy()
            if "bok_rate" in macro_features.columns:
                macro_features["bok_rate_change_3m"] = macro_features["bok_rate"].diff(3)
            if "m2" in macro_features.columns:
                macro_features["m2_yoy"] = macro_features["m2"].pct_change(12) * 100
            panel = panel.merge(macro_features, on=["ym", "date"], how="left")

        panel = panel.sort_values(["aptSeq", "date"]).reset_index(drop=True)
        self._write_output_parquet("complex_monthly_panel.parquet", panel)
        return panel

    def build_complex_forecast_targets(self, complex_panel: pd.DataFrame | None = None) -> pd.DataFrame:
        """단지 예측용 타깃/래그 패널을 생성한다."""
        complex_panel = complex_panel if complex_panel is not None else self._read_output_parquet("complex_monthly_panel.parquet")
        if complex_panel.empty:
            return pd.DataFrame()

        panel = complex_panel.sort_values(["aptSeq", "date"]).copy()
        lag_columns = [
            "trade_price_per_m2",
            "trade_price_std84",
            "jeonse_deposit_per_m2",
            "jeonse_deposit_std84",
            "wolse_monthly_rent_per_m2",
            "trade_count",
            "jeonse_ratio",
            "conversion_rate",
        ]
        lag_columns = [column for column in lag_columns if column in panel.columns]

        grouped = panel.groupby("aptSeq", observed=True)
        for lag in [1, 3, 6, 12]:
            for column in lag_columns:
                panel[f"{column}_lag{lag}"] = grouped[column].shift(lag)

        short_term_targets = {
            "trade_price_per_m2": [1, 3],
            "trade_price_std84": [1, 3],
            "jeonse_deposit_per_m2": [1, 3],
            "jeonse_deposit_std84": [1, 3],
            "wolse_monthly_rent_per_m2": [1, 3],
            "jeonse_ratio": [1, 3],
            "conversion_rate": [1, 3],
        }
        for column, horizons in short_term_targets.items():
            if column not in panel.columns:
                continue
            for horizon in horizons:
                panel[f"{column}_t{horizon}"] = grouped[column].shift(-horizon)

        for base_col, target_col in [
            ("trade_price_per_m2", "future_trade_return_12m"),
            ("jeonse_deposit_per_m2", "future_jeonse_return_12m"),
            ("wolse_monthly_rent_per_m2", "future_wolse_return_12m"),
        ]:
            if base_col in panel.columns:
                future = grouped[base_col].shift(-12)
                panel[target_col] = (future / panel[base_col] - 1) * 100

        panel = panel.sort_values(["aptSeq", "date"]).reset_index(drop=True)
        self._write_output_parquet("complex_forecast_targets.parquet", panel)
        return panel

    def build_representative_datasets(
        self,
        complex_master: pd.DataFrame | None = None,
        macro_monthly: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """59형/84형 대표단지 분석용 선계산 parquet를 생성한다."""
        pipeline = RepresentativeComplexPipeline(output_dir=self.output_dir)
        return pipeline.run_all(
            complex_master=complex_master if complex_master is not None else self._read_output_parquet("complex_master.parquet"),
            macro_monthly=macro_monthly if macro_monthly is not None else self._read_output_parquet("macro_monthly.parquet"),
        )

    def run_all(self) -> None:
        """매매 집계, 전월세 집계, 거시지표 통합, 대시보드 데이터셋 생성을 순차 실행한다."""
        trade_summary = self.build_monthly_trade_summary()
        rent_summary = self.build_monthly_rent_summary()
        macro_monthly = self.build_macro_monthly()
        self.build_dashboard_datasets(trade_summary, rent_summary, macro_monthly)
        complex_master = self.build_complex_master()
        complex_panel = self.build_complex_monthly_panel(complex_master=complex_master, macro_monthly=macro_monthly)
        self.build_complex_forecast_targets(complex_panel)
        self.build_representative_datasets(complex_master=complex_master, macro_monthly=macro_monthly)
        logger.info("전체 집계 파이프라인 완료")


if __name__ == "__main__":
    pipeline = AggregationPipeline()
    pipeline.run_all()
