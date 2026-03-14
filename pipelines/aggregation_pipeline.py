"""월별 집계 파이프라인.

Raw 데이터를 읽어 월별 × 지역별 요약 통계와 대시보드 전용 경량 parquet를 생성한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    ALL_REGIONS,
    ECOS_RAW_DIR,
    MARKET_RAW_DIR,
    MOLIT_RAW_DIR,
    PROCESSED_DIR,
)


class AggregationPipeline:
    """Raw 데이터를 집계하여 processed 파일을 생성하는 파이프라인."""

    AREA_BINS: list[float] = [0, 60, 85, float("inf")]
    AREA_LABELS: list[str] = ["60㎡이하", "60~85㎡", "85㎡초과"]
    DASHBOARD_START_YM: str = "202001"
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

    def _load_parquet_batch(self, directory: Path) -> pd.DataFrame:
        """디렉토리 내 모든 parquet 파일을 읽어 합친다."""
        files = sorted(directory.glob("*.parquet"))
        if not files:
            logger.warning(f"파일 없음: {directory}")
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                parts = file_path.stem.split("_")
                if len(parts) >= 2:
                    df["_lawd_cd"] = parts[0]
                    df["_deal_ym"] = parts[1]
                    df["_region_name"] = ALL_REGIONS.get(parts[0], parts[0])
                frames.append(df)
            except Exception as exc:
                logger.warning(f"파일 로드 실패: {file_path.name} - {exc}")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

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

        df["area_group"] = pd.cut(
            df["area"],
            bins=self.AREA_BINS,
            labels=self.AREA_LABELS,
            right=True,
        )

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

        trim_means = (
            df.groupby(group_cols, observed=True)["price"]
            .apply(self._trimmed_mean)
            .reset_index(name="절사평균거래금액")
        )
        summary = summary.merge(trim_means, on=group_cols, how="left")

        avg_area = (
            df.groupby(group_cols, observed=True)["area"]
            .mean()
            .reset_index(name="평균전용면적")
        )
        summary = summary.merge(avg_area, on=group_cols, how="left")

        avg_age = (
            df.groupby(group_cols, observed=True)["age"]
            .mean()
            .reset_index(name="평균건물연령")
        )
        summary = summary.merge(avg_age, on=group_cols, how="left")

        area_avg = (
            df.groupby(group_cols + ["area_group"], observed=True)["price"]
            .mean()
            .reset_index()
            .pivot_table(
                index=group_cols,
                columns="area_group",
                values="price",
                observed=True,
            )
            .reset_index()
        )
        area_avg.columns = [
            f"평균거래금액_{column}" if column in self.AREA_LABELS else column
            for column in area_avg.columns
        ]
        summary = summary.merge(area_avg, on=group_cols, how="left")

        out_path = self.output_dir / "monthly_trade_summary.parquet"
        summary.to_parquet(out_path, index=False)
        logger.info(f"매매 집계 저장 완료: {out_path.name} ({len(summary)}건)")
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

        group_cols = ["ym", "date", "_lawd_cd", "_region_name", "rentType"]
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

        out_path = self.output_dir / "monthly_rent_summary.parquet"
        summary.to_parquet(out_path, index=False)
        logger.info(f"전월세 집계 저장 완료: {out_path.name} ({len(summary)}건)")
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
        out_path = self.output_dir / "macro_monthly.parquet"
        merged.to_parquet(out_path, index=False)
        logger.info(f"거시지표 통합 저장 완료: {out_path.name} ({len(merged)}건)")
        return merged

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
            start_date = pd.Timestamp(f"{self.DASHBOARD_START_YM[:4]}-{self.DASHBOARD_START_YM[4:]}-01")
            return result[result[date_col] >= start_date].copy()

        return result

    def _read_output_parquet(self, name: str) -> pd.DataFrame:
        """output_dir 아래 parquet를 읽는다."""
        path = self.output_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def build_dashboard_datasets(
        self,
        trade_summary: pd.DataFrame | None = None,
        rent_summary: pd.DataFrame | None = None,
        macro_monthly: pd.DataFrame | None = None,
    ) -> None:
        """대시보드 전용 경량 parquet 세트를 생성한다."""
        logger.info("대시보드 전용 데이터셋 생성 시작")

        trade_summary = trade_summary if trade_summary is not None else self._read_output_parquet("monthly_trade_summary.parquet")
        rent_summary = rent_summary if rent_summary is not None else self._read_output_parquet("monthly_rent_summary.parquet")
        macro_monthly = macro_monthly if macro_monthly is not None else self._read_output_parquet("macro_monthly.parquet")

        dashboard_trade_summary = self._filter_dashboard_window(trade_summary, ym_col="ym")
        dashboard_rent_summary = self._filter_dashboard_window(rent_summary, ym_col="ym")
        dashboard_macro = self._filter_dashboard_window(macro_monthly, date_col="date")

        dashboard_trade_summary.to_parquet(self.output_dir / "dashboard_trade_summary.parquet", index=False)
        dashboard_rent_summary.to_parquet(self.output_dir / "dashboard_rent_summary.parquet", index=False)
        dashboard_macro.to_parquet(self.output_dir / "dashboard_macro_monthly.parquet", index=False)

        trade_detail = self._load_processed_chunks("apt_trade", columns=self.DASHBOARD_TRADE_DETAIL_COLUMNS)
        trade_detail = self._filter_dashboard_window(trade_detail, date_col="date")
        trade_detail.to_parquet(self.output_dir / "dashboard_trade_detail.parquet", index=False)

        rent_detail = self._load_processed_chunks("apt_rent", columns=self.DASHBOARD_RENT_DETAIL_COLUMNS)
        if not rent_detail.empty and "rentType" not in rent_detail.columns:
            rent_detail["rentType"] = np.where(rent_detail["monthly_rent"].fillna(0) == 0, "전세", "월세")
        rent_detail = self._filter_dashboard_window(rent_detail, date_col="date")
        rent_detail.to_parquet(self.output_dir / "dashboard_rent_detail.parquet", index=False)

        logger.info(
            "대시보드 데이터셋 저장 완료: "
            f"trade_summary={len(dashboard_trade_summary)}, "
            f"rent_summary={len(dashboard_rent_summary)}, "
            f"macro={len(dashboard_macro)}, "
            f"trade_detail={len(trade_detail)}, "
            f"rent_detail={len(rent_detail)}"
        )

    def run_all(self) -> None:
        """매매 집계, 전월세 집계, 거시지표 통합, 대시보드 데이터셋 생성을 순차 실행한다."""
        trade_summary = self.build_monthly_trade_summary()
        rent_summary = self.build_monthly_rent_summary()
        macro_monthly = self.build_macro_monthly()
        self.build_dashboard_datasets(trade_summary, rent_summary, macro_monthly)
        logger.info("전체 집계 파이프라인 완료")


if __name__ == "__main__":
    pipeline = AggregationPipeline()
    pipeline.run_all()
