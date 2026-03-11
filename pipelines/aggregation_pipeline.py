"""월별 집계 파이프라인.

Raw 데이터를 읽어 월별 × 지역별 요약 통계를 생성한다.
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
    """Raw 데이터를 집계하여 processed 파일을 생성하는 파이프라인.

    Attributes:
        molit_dir: 국토부 Raw 데이터 디렉토리.
        ecos_dir: ECOS Raw 데이터 디렉토리.
        market_dir: yfinance Raw 데이터 디렉토리.
        output_dir: 집계 결과 저장 디렉토리.
    """

    # 면적 구간 경계
    AREA_BINS: list[float] = [0, 60, 85, float("inf")]
    AREA_LABELS: list[str] = ["60㎡이하", "60~85㎡", "85㎡초과"]

    def __init__(
        self,
        molit_dir: str | Path | None = None,
        ecos_dir: str | Path | None = None,
        market_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """AggregationPipeline을 초기화한다.

        Args:
            molit_dir: 국토부 Raw 데이터 경로.
            ecos_dir: ECOS Raw 데이터 경로.
            market_dir: yfinance Raw 데이터 경로.
            output_dir: 집계 결과 저장 경로.
        """
        self.molit_dir = Path(molit_dir) if molit_dir else MOLIT_RAW_DIR
        self.ecos_dir = Path(ecos_dir) if ecos_dir else ECOS_RAW_DIR
        self.market_dir = Path(market_dir) if market_dir else MARKET_RAW_DIR
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _trimmed_mean(series: pd.Series, trim_pct: float = 0.1) -> float:
        """상하위 일정 비율을 절사한 평균을 계산한다.

        Args:
            series: 수치 시리즈.
            trim_pct: 절사 비율 (기본 10%).

        Returns:
            절사평균 값.
        """
        s = series.dropna().sort_values()
        if len(s) == 0:
            return np.nan
        n = len(s)
        trim_count = int(n * trim_pct)
        if trim_count > 0:
            s = s.iloc[trim_count:-trim_count]
        return float(s.mean()) if len(s) > 0 else np.nan

    def _load_parquet_batch(self, directory: Path) -> pd.DataFrame:
        """디렉토리 내 모든 parquet 파일을 배치로 읽어 합친다.

        Args:
            directory: parquet 파일들이 있는 디렉토리.

        Returns:
            모든 파일을 합친 DataFrame.
        """
        files = sorted(directory.glob("*.parquet"))
        if not files:
            logger.warning(f"파일 없음: {directory}")
            return pd.DataFrame()

        dfs: list[pd.DataFrame] = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                # 파일명에서 지역코드와 연월 추출
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    df["_lawd_cd"] = parts[0]
                    df["_deal_ym"] = parts[1]
                    df["_region_name"] = ALL_REGIONS.get(parts[0], parts[0])
                dfs.append(df)
            except Exception as exc:
                logger.warning(f"파일 로드 실패: {f.name} – {exc}")

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _load_processed_chunks(self, prefix: str) -> pd.DataFrame:
        """prefix로 시작하는 모든 전처리된 parquet 조각을 읽어 합친다.

        Args:
            prefix: 'apt_trade' 또는 'apt_rent'.

        Returns:
            합쳐진 DataFrame.
        """
        files = sorted(self.output_dir.glob(f"{prefix}_*.parquet"))
        if not files:
            # 조각 파일이 없으면 전체 파일 시도 (하위 호환)
            full_path = self.output_dir / f"{prefix}.parquet"
            if full_path.exists():
                return pd.read_parquet(full_path)
            logger.warning(f"전처리된 {prefix} 데이터 조각을 찾을 수 없습니다.")
            return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as exc:
                logger.error(f"조각 로드 실패: {f.name} - {exc}")

        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)

    def build_monthly_trade_summary(self) -> pd.DataFrame:
        """전처리된 매매 데이터를 로드하여 월별 × 지역별 집계를 생성한다.

        Returns:
            월별 매매 집계 DataFrame.
        """
        logger.info("매매 월별 집계 시작 (전처리 데이터 조각 사용)")
        df = self._load_processed_chunks("apt_trade")
        
        if df.empty:
            return pd.DataFrame()

        # 집계 연월 추출 (YYYYMM)
        df["ym"] = df["date"].dt.strftime("%Y%m")

        # 지역 코드 추출 (dong_repr 에 포함된 괄호 안의 코드 활용)
        # 예: "신림동(11620)" -> "11620"
        df["_lawd_cd"] = df["dong_repr"].str.extract(r"\((\d+)\)")
        df["_region_name"] = df["dong_repr"].str.split("(").str[0]

        # 이상치 제거 (상하위 1%) - 전처리 단계에서 수행하지 않았다면 여기서 수행
        q_low = df["price"].quantile(0.01)
        q_high = df["price"].quantile(0.99)
        df = df[(df["price"] >= q_low) & (df["price"] <= q_high)].copy()

        # 면적 구간 파생 (분석용 area 사용)
        df["area_group"] = pd.cut(
            df["area"],
            bins=self.AREA_BINS,
            labels=self.AREA_LABELS,
            right=True,
        )

        # 그룹 키
        group_cols = ["ym", "_lawd_cd", "_region_name"]

        # 기본 집계
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

        # 절사평균
        trim_means = (
            df.groupby(group_cols, observed=True)["price"]
            .apply(self._trimmed_mean)
            .reset_index(name="절사평균거래금액")
        )
        summary = summary.merge(trim_means, on=group_cols, how="left")

        # 평균 전용면적
        avg_area = (
            df.groupby(group_cols, observed=True)["area"]
            .mean()
            .reset_index(name="평균전용면적")
        )
        summary = summary.merge(avg_area, on=group_cols, how="left")

        # 평균 건물 연령 (전처리된 age 사용)
        avg_age = (
            df.groupby(group_cols, observed=True)["age"]
            .mean()
            .reset_index(name="평균건물연령")
        )
        summary = summary.merge(avg_age, on=group_cols, how="left")

        # 면적대별 평균 거래금액
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
            f"평균거래금액_{c}" if c in self.AREA_LABELS else c
            for c in area_avg.columns
        ]
        summary = summary.merge(area_avg, on=group_cols, how="left")

        # 저장
        out_path = self.output_dir / "monthly_trade_summary.parquet"
        summary.to_parquet(out_path, index=False)
        logger.info(f"매매 집계 저장 완료: {out_path.name} ({len(summary)}건)")
        return summary

    def build_monthly_rent_summary(self) -> pd.DataFrame:
        """전처리된 전월세 데이터를 로드하여 월별 × 지역별 집계를 생성한다.

        Returns:
            월별 전월세 집계 DataFrame.
        """
        logger.info("전월세 월별 집계 시작 (전처리 데이터 조각 사용)")
        df = self._load_processed_chunks("apt_rent")
        
        if df.empty:
            return pd.DataFrame()

        df["ym"] = df["date"].dt.strftime("%Y%m")
        df["_lawd_cd"] = df["dong_repr"].str.extract(r"\((\d+)\)")
        df["_region_name"] = df["dong_repr"].str.split("(").str[0]

        # rentType을 반드시 포함하여 집계
        group_cols = ["ym", "date", "_lawd_cd", "_region_name", "rentType"]

        # 만약 전처리 과정에서 rentType이 생성되지 않았다면 (비정상 케이스 대비)
        if "rentType" not in df.columns:
            df["rentType"] = np.where(df["monthly_rent"] == 0, "전세", "월세")

        agg_parts = {
            "거래건수": ("deposit", "size"),
            "평균보증금": ("deposit", "mean"),
            "중앙값보증금": ("deposit", "median"),
            "평균월세": ("monthly_rent", "mean"),
            "중앙값월세": ("monthly_rent", "median"),
            "평균84환산보증금": ("deposit_std84", "mean"),
        }

        summary = (
            df.groupby(group_cols, observed=True)
            .agg(**agg_parts)
            .reset_index()
        )

        # 저장
        out_path = self.output_dir / "monthly_rent_summary.parquet"
        summary.to_parquet(out_path, index=False)
        logger.info(f"전월세 집계 저장 완료: {out_path.name} ({len(summary)}건)")
        return summary

    def build_macro_monthly(self) -> pd.DataFrame:
        """ECOS + yfinance Raw 데이터를 월별 통합 테이블로 병합한다.

        Returns:
            거시지표 월별 통합 DataFrame.
        """
        logger.info("거시지표 월별 통합 시작")
        merged: pd.DataFrame | None = None

        # ECOS 데이터 로드
        for f in sorted(self.ecos_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                label = f.stem
                df = df[["date", "value"]].rename(columns={"value": label})
                if merged is None:
                    merged = df
                else:
                    merged = merged.merge(df, on="date", how="outer")
            except Exception as exc:
                logger.warning(f"ECOS 로드 실패: {f.name} – {exc}")

        # yfinance 데이터 로드
        for f in sorted(self.market_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                label = f.stem
                df["date"] = pd.to_datetime(df["date"])
                # 월의 첫 날로 정규화 (ECOS와 조인 위해)
                df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
                df = df[["date", "close"]].rename(columns={"close": label})
                if merged is None:
                    merged = df
                else:
                    merged = merged.merge(df, on="date", how="outer")
            except Exception as exc:
                logger.warning(f"yfinance 로드 실패: {f.name} – {exc}")

        if merged is None or merged.empty:
            logger.warning("거시지표 데이터가 없어 집계를 건너뜁니다.")
            return pd.DataFrame()

        merged = merged.sort_values("date").reset_index(drop=True)

        # 저장
        out_path = self.output_dir / "macro_monthly.parquet"
        merged.to_parquet(out_path, index=False)
        logger.info(f"거시지표 통합 저장 완료: {out_path.name} ({len(merged)}건)")
        return merged

    def run_all(self) -> None:
        """매매 집계, 전월세 집계, 거시지표 통합을 순차 실행한다."""
        self.build_monthly_trade_summary()
        self.build_monthly_rent_summary()
        self.build_macro_monthly()
        logger.info("전체 집계 파이프라인 완료")


if __name__ == "__main__":
    pipeline = AggregationPipeline()
    pipeline.run_all()
