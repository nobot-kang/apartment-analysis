"""yfinance 기반 시장 데이터 수집 파이프라인.

금 선물, WTI 원유, 원달러 환율 등의 월말 종가를 수집한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from config.settings import MARKET_RAW_DIR, YFINANCE_TICKERS


class MarketPipeline:
    """yfinance 시장 데이터 수집 파이프라인.

    Attributes:
        save_dir: Raw 데이터 저장 디렉토리.
    """

    def __init__(self, save_dir: str | Path | None = None) -> None:
        """MarketPipeline을 초기화한다.

        Args:
            save_dir: Raw 데이터 저장 경로. 기본값은 ``data/raw/market``.
        """
        self.save_dir = Path(save_dir) if save_dir else MARKET_RAW_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def fetch_yfinance(
        self,
        ticker: str,
        label: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """yfinance에서 일별 데이터를 가져와 월말 종가로 리샘플링한다.

        Args:
            ticker: yfinance 티커 심볼 (예: ``GC=F``).
            label: 저장 파일명/지표 라벨.
            start: 조회 시작일 ``YYYY-MM-DD``.
            end: 조회 종료일 ``YYYY-MM-DD``.

        Returns:
            ``date``, ``close``, ``ticker`` 컬럼을 포함하는 월별 DataFrame.
        """
        logger.info(f"yfinance 수집 시작: {label} ({ticker})")

        raw = yf.download(ticker, start=start, end=end, progress=False)

        if raw.empty:
            logger.warning(f"yfinance 데이터 없음: {label} ({ticker})")
            return pd.DataFrame()

        # MultiIndex 컬럼 처리 (yfinance가 가끔 MultiIndex로 반환)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # 월말 종가 리샘플링
        monthly = raw[["Close"]].resample("ME").last().dropna()
        monthly = monthly.reset_index()
        monthly.columns = ["date", "close"]
        monthly["ticker"] = ticker

        # 저장
        out_path = self.save_dir / f"{label}.parquet"
        monthly.to_parquet(out_path, index=False)
        logger.info(f"yfinance 저장 완료: {out_path.name} ({len(monthly)}건)")

        return monthly

    def run_all(self, start: str, end: str) -> dict[str, pd.DataFrame]:
        """설정에 정의된 모든 yfinance 티커를 순차 수집한다.

        Args:
            start: 조회 시작일 ``YYYY-MM-DD``.
            end: 조회 종료일 ``YYYY-MM-DD``.

        Returns:
            ``{label: DataFrame}`` 딕셔너리.
        """
        results: dict[str, pd.DataFrame] = {}

        for name, info in YFINANCE_TICKERS.items():
            df = self.fetch_yfinance(
                ticker=info["ticker"],
                label=name,
                start=start,
                end=end,
            )
            results[name] = df

        logger.info(f"yfinance 전체 수집 완료: {len(results)}개 지표")
        return results
