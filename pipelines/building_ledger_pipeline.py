"""건축물대장 표제부 수집 파이프라인.

수집된 매매/전월세 데이터의 고유 아파트 목록을 기반으로
건축물대장 표제부(getBrTitleInfo) 정보를 수집하여 parquet 파일로 저장한다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import xmltodict
from loguru import logger
from tqdm import tqdm

from config.settings import (
    BUILDING_LEDGER_ENDPOINT,
    MOLIT_API_SLEEP,
    MOLIT_RAW_DIR,
    get_api_key,
)
from pipelines.apartment_list import ApartmentListManager


class BuildingLedgerPipeline:
    """건축물대장 표제부(getBrTitleInfo) API 수집 파이프라인.

    ``ApartmentListManager`` 에서 가져온 고유 아파트 목록을 기반으로
    건축물대장 표제부 정보를 수집한다.

    Attributes:
        api_key: 공공데이터포털 서비스 키.
        save_dir: 건축물대장 Raw 데이터 저장 디렉토리.
    """

    def __init__(
        self,
        api_key: str | None = None,
        save_dir: str | Path | None = None,
        molit_dir: str | Path | None = None,
    ) -> None:
        """BuildingLedgerPipeline을 초기화한다.

        Args:
            api_key: 공공데이터포털 서비스 키. ``None`` 이면 환경변수에서 로드.
            save_dir: 건축물대장 저장 경로. 기본값은 ``data/raw/molit/building_ledger``.
            molit_dir: 국토부 Raw 루트 경로 (아파트 목록 로드용).
        """
        self.api_key = api_key or get_api_key("MOLIT_API_KEY")
        self.save_dir = Path(save_dir) if save_dir else MOLIT_RAW_DIR / "building_ledger"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.apt_manager = ApartmentListManager(
            molit_dir=Path(molit_dir) if molit_dir else MOLIT_RAW_DIR,
        )

    def _fetch_single(
        self,
        sigungu_cd: str,
        bjdong_cd: str,
        bun: str,
        ji: str,
    ) -> pd.DataFrame:
        """단일 지번의 건축물대장 표제부를 조회한다.

        Args:
            sigungu_cd: 시군구코드 5자리.
            bjdong_cd: 법정동코드 5자리.
            bun: 본번 4자리.
            ji: 부번 4자리.

        Returns:
            건축물대장 DataFrame. 데이터 없으면 빈 DataFrame.
        """
        params = {
            "serviceKey": self.api_key,
            "sigunguCd": sigungu_cd,
            "bjdongCd": bjdong_cd,
            "bun": str(bun).zfill(4),
            "ji": str(ji).zfill(4),
            "numOfRows": 100,
            "pageNo": 1,
        }

        resp = requests.get(BUILDING_LEDGER_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        return self._parse_response(resp.text)

    @staticmethod
    def _parse_response(xml_text: str) -> pd.DataFrame:
        """XML 응답을 파싱하여 DataFrame으로 반환한다.

        원본 컬럼을 모두 보존한다.

        Args:
            xml_text: API 응답 XML 문자열.

        Returns:
            파싱된 DataFrame.
        """
        data_dict = xmltodict.parse(xml_text)

        if "response" not in data_dict:
            return pd.DataFrame()

        response = data_dict["response"]
        header = response.get("header", {})
        result_code = header.get("resultCode")

        if result_code not in ("00", "000"):
            return pd.DataFrame()

        body = response.get("body", {})
        items = body.get("items")

        if not items or items.get("item") is None:
            return pd.DataFrame()

        item_list = items["item"]
        if isinstance(item_list, dict):
            item_list = [item_list]

        df = pd.DataFrame(item_list)

        # regstrGbCd == "2" : 집합건물 (아파트 등)
        if "regstrGbCd" in df.columns:
            filtered = df[df["regstrGbCd"] == "2"]
            if not filtered.empty:
                return filtered
        return df

    def fetch_for_apartment(self, apt_row: pd.Series) -> pd.DataFrame:
        """아파트 목록의 단일 행으로 건축물대장을 조회한다.

        Args:
            apt_row: ``sggCd``, ``umdCd``, ``bonbun``, ``bubun``, ``aptSeq``
                     컬럼을 포함하는 Series.

        Returns:
            건축물대장 DataFrame (``aptSeq`` 컬럼 추가).
        """
        df = self._fetch_single(
            sigungu_cd=str(apt_row["sggCd"]),
            bjdong_cd=str(apt_row["umdCd"]),
            bun=str(apt_row["bonbun"]),
            ji=str(apt_row["bubun"]),
        )
        if not df.empty:
            df["aptSeq"] = apt_row["aptSeq"]
        return df

    def run_full_collection(self, force_rebuild_list: bool = False) -> None:
        """수집된 매매/전월세의 고유 아파트 전체에 대해 건축물대장을 수집한다.

        이미 저장된 ``aptSeq`` 는 건너뛴다 (증분 수집).
        API rate limit 대비 연속 실패 시 대기 시간을 점진적으로 늘린다.

        Args:
            force_rebuild_list: ``True`` 이면 아파트 목록을 강제 재생성.
        """
        # 아파트 목록 준비
        if force_rebuild_list or not self.apt_manager.list_path.exists():
            self.apt_manager.build_list()

        apt_params = self.apt_manager.get_building_ledger_params()
        if apt_params.empty:
            logger.warning("건축물대장 조회 가능한 아파트가 없습니다.")
            return

        # 이미 수집된 aptSeq 파악
        collected_seqs: set[str] = set()
        existing_files = list(self.save_dir.glob("*.parquet"))
        for f in existing_files:
            try:
                existing = pd.read_parquet(f, columns=["aptSeq"])
                collected_seqs.update(existing["aptSeq"].unique())
            except Exception:
                pass

        # 미수집 목록
        pending = apt_params[~apt_params["aptSeq"].isin(collected_seqs)]
        logger.info(
            f"건축물대장 수집 대상: {len(pending)}건 "
            f"(전체 {len(apt_params)}, 수집완료 {len(collected_seqs)})"
        )

        if pending.empty:
            logger.info("모든 건축물대장이 수집 완료되었습니다.")
            return

        failed: list[dict[str, str]] = []
        consecutive_errors = 0
        max_consecutive_errors = 10

        # 배치 단위로 저장 (100건마다)
        batch: list[pd.DataFrame] = []
        batch_count = 0
        batch_size = 100

        for _, row in tqdm(pending.iterrows(), total=len(pending), desc="건축물대장 수집"):
            apt_seq = row["aptSeq"]

            try:
                df = self.fetch_for_apartment(row)

                if not df.empty:
                    batch.append(df)
                    batch_count += 1

                consecutive_errors = 0
                time.sleep(MOLIT_API_SLEEP)

                # 배치 저장
                if batch_count >= batch_size:
                    self._save_batch(batch)
                    batch = []
                    batch_count = 0

            except Exception as exc:
                consecutive_errors += 1
                logger.warning(
                    f"건축물대장 수집 실패 ({apt_seq}): {exc}"
                )
                failed.append({
                    "aptSeq": apt_seq,
                    "sggCd": str(row["sggCd"]),
                    "umdCd": str(row["umdCd"]),
                    "bonbun": str(row["bonbun"]),
                    "bubun": str(row["bubun"]),
                    "error": str(exc),
                })

                if consecutive_errors >= max_consecutive_errors:
                    wait_time = min(60, consecutive_errors * 5)
                    logger.warning(
                        f"연속 {consecutive_errors}회 실패 – "
                        f"{wait_time}초 대기 후 재개"
                    )
                    time.sleep(wait_time)

                    if consecutive_errors >= max_consecutive_errors * 3:
                        logger.error(
                            f"연속 {consecutive_errors}회 실패 – 수집 중단. "
                            f"failed_list_building.json 확인 후 재실행하세요."
                        )
                        break
                else:
                    time.sleep(MOLIT_API_SLEEP * 2)

        # 남은 배치 저장
        if batch:
            self._save_batch(batch)

        # 실패 목록 저장
        failed_path = self.save_dir / "failed_list_building.json"
        if failed:
            existing_failed: list[dict[str, str]] = []
            if failed_path.exists():
                existing_failed = json.loads(
                    failed_path.read_text(encoding="utf-8")
                )
            merged = existing_failed + failed
            failed_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.warning(f"실패 {len(failed)}건 기록됨: {failed_path}")
        else:
            logger.info("건축물대장 수집 완료 (실패 없음)")

    def _save_batch(self, batch: list[pd.DataFrame]) -> None:
        """수집된 배치를 하나의 parquet 파일로 저장한다.

        Args:
            batch: DataFrame 리스트.
        """
        if not batch:
            return

        combined = pd.concat(batch, ignore_index=True)

        # 기존 누적 파일에 append
        cumulative_path = self.save_dir / "building_ledger_all.parquet"
        if cumulative_path.exists():
            existing = pd.read_parquet(cumulative_path)
            combined = pd.concat([existing, combined], ignore_index=True)
            # aptSeq 기준 중복 제거 (최신 우선)
            combined = combined.drop_duplicates(subset=["aptSeq"], keep="last")

        combined.to_parquet(cumulative_path, index=False)
        logger.info(f"건축물대장 배치 저장: {len(combined)}건 → {cumulative_path.name}")

    def retry_failed(self) -> None:
        """``failed_list_building.json`` 의 실패 항목을 재시도한다."""
        failed_path = self.save_dir / "failed_list_building.json"
        if not failed_path.exists():
            logger.info("재시도할 실패 목록 없음")
            return

        items: list[dict[str, str]] = json.loads(
            failed_path.read_text(encoding="utf-8")
        )
        still_failed: list[dict[str, str]] = []
        batch: list[pd.DataFrame] = []

        for item in tqdm(items, desc="건축물대장 재시도"):
            try:
                row = pd.Series(item)
                df = self.fetch_for_apartment(row)
                if not df.empty:
                    batch.append(df)
                time.sleep(MOLIT_API_SLEEP)
            except Exception as exc:
                logger.warning(f"재시도 실패 ({item['aptSeq']}): {exc}")
                item["error"] = str(exc)
                still_failed.append(item)
                time.sleep(MOLIT_API_SLEEP * 2)

        if batch:
            self._save_batch(batch)

        if still_failed:
            failed_path.write_text(
                json.dumps(still_failed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.warning(f"재시도 후에도 {len(still_failed)}건 실패")
        else:
            failed_path.unlink(missing_ok=True)
            logger.info("모든 재시도 성공")
