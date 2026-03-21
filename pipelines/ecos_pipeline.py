"""한국은행 ECOS API 수집 파이프라인.

기준금리, CPI, M2 등 거시경제 지표를 수집하여 parquet 파일로 저장한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger

from config.settings import (
    ECOS_BASE_URL,
    ECOS_INDICATORS,
    ECOS_RAW_DIR,
)


class EcosPipeline:
    """한국은행 ECOS 통계 수집 파이프라인.

    Attributes:
        api_key: 한국은행 ECOS에서 발급받은 API 키.
        save_dir: Raw 데이터 저장 디렉토리.
    """

    # ECOS API는 한 번에 최대 100건 조회 가능
    MAX_ROWS_PER_REQUEST: int = 100

    def __init__(self, api_key: str, save_dir: str | Path | None = None) -> None:
        """EcosPipeline을 초기화한다.

        Args:
            api_key: ECOS API 키.
            save_dir: Raw 데이터 저장 경로. 기본값은 ``data/raw/ecos``.
        """
        self.api_key = api_key
        self.save_dir = Path(save_dir) if save_dir else ECOS_RAW_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def fetch_statistic(
        self,
        stat_code: str,
        item_code: str,
        start_ym: str,
        end_ym: str,
        label: str,
    ) -> pd.DataFrame:
        """ECOS API에서 단일 통계 지표를 수집한다.

        페이지네이션을 자동 처리하여 전체 데이터를 가져온다.

        Args:
            stat_code: 통계표 코드 (예: ``722Y001``).
            item_code: 항목 코드 (예: ``0101000``).
            start_ym: 조회 시작 연월 ``YYYYMM``.
            end_ym: 조회 종료 연월 ``YYYYMM``.
            label: 저장 파일명/지표 라벨.

        Returns:
            ``date``, ``value`` 등 컬럼을 포함하는 DataFrame.
        """
        all_rows: list[dict[str, Any]] = []
        start_idx = 1

        while True:
            end_idx = start_idx + self.MAX_ROWS_PER_REQUEST - 1
            url = (
                f"{ECOS_BASE_URL}/{self.api_key}/json/kr/"
                f"{start_idx}/{end_idx}/{stat_code}/M/{start_ym}/{end_ym}/{item_code}"
            )

            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # 에러 처리
            if "StatisticSearch" not in data:
                error_msg = data.get("RESULT", {}).get("MESSAGE", "알 수 없는 오류")
                logger.warning(f"ECOS API 응답 오류 ({label}): {error_msg}")
                break

            rows = data["StatisticSearch"].get("row", [])
            all_rows.extend(rows)

            total_count = int(data["StatisticSearch"].get("list_total_count", 0))

            # 모든 데이터를 가져왔으면 종료
            if start_idx + len(rows) - 1 >= total_count:
                break

            start_idx += self.MAX_ROWS_PER_REQUEST

        if not all_rows:
            logger.warning(f"ECOS 데이터 없음: {label}")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # 컬럼 정리
        df = df.rename(columns={
            "TIME": "date_str",
            "DATA_VALUE": "value",
            "UNIT_NAME": "unit",
            "STAT_NAME": "stat_name",
        })

        df["date"] = pd.to_datetime(df["date_str"], format="%Y%m")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value", "unit", "stat_name"]].sort_values("date").reset_index(drop=True)

        # 저장
        out_path = self.save_dir / f"{label}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"ECOS 저장 완료: {out_path.name} ({len(df)}건)")

        return df

    def run_all(self, start_ym: str, end_ym: str) -> dict[str, pd.DataFrame]:
        """설정에 정의된 모든 ECOS 지표를 순차 수집한다.

        Args:
            start_ym: 조회 시작 연월 ``YYYYMM``.
            end_ym: 조회 종료 연월 ``YYYYMM``.

        Returns:
            ``{label: DataFrame}`` 딕셔너리.
        """
        results: dict[str, pd.DataFrame] = {}

        for name, info in ECOS_INDICATORS.items():
            logger.info(f"ECOS 수집 시작: {info['label']} ({name})")
            df = self.fetch_statistic(
                stat_code=info["stat_code"],
                item_code=info["item_code"],
                start_ym=start_ym,
                end_ym=end_ym,
                label=name,
            )
            results[name] = df

        logger.info(f"ECOS 전체 수집 완료: {len(results)}개 지표")
        return results
