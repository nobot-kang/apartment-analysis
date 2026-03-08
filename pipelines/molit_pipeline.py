"""국토교통부 실거래가 Open API 수집 파이프라인.

아파트 매매 및 전월세 실거래가를 법정동코드 × 월 단위로 수집하여
parquet 파일로 저장한다.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from config.settings import (
    ALL_REGIONS,
    MOLIT_API_SLEEP,
    MOLIT_APT_RENT_ENDPOINT,
    MOLIT_APT_TRADE_ENDPOINT,
    MOLIT_NUM_OF_ROWS,
    MOLIT_RAW_DIR,
)


class MolitPipeline:
    """국토부 실거래가 API 수집 파이프라인.

    법정동코드와 거래연월을 조합하여 아파트 매매/전월세 데이터를
    수집하고, parquet 형식으로 로컬에 저장한다.

    Attributes:
        api_key: 공공데이터포털에서 발급받은 서비스 키.
        save_dir: Raw 데이터 저장 루트 디렉토리.
    """

    def __init__(self, api_key: str, save_dir: str | Path | None = None) -> None:
        """MolitPipeline을 초기화한다.

        Args:
            api_key: 공공데이터포털 서비스 키.
            save_dir: Raw 데이터 저장 경로. 기본값은 ``data/raw/molit``.
        """
        self.api_key = api_key
        self.save_dir = Path(save_dir) if save_dir else MOLIT_RAW_DIR

        # 하위 디렉토리 보장
        (self.save_dir / "apt_trade").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "apt_rent").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xml_response(text: str) -> list[dict[str, Any]]:
        """XML 응답을 파싱하여 item 목록을 반환한다.

        Args:
            text: API 응답 본문 (XML).

        Returns:
            각 ``<item>`` 을 딕셔너리로 변환한 리스트.

        Raises:
            RuntimeError: 응답 코드가 정상이 아닐 때.
        """
        root = ET.fromstring(text)

        # 결과 코드 확인 (공공데이터포털: "00" 또는 "000" = 정상)
        result_code = root.findtext(".//resultCode")
        result_msg = root.findtext(".//resultMsg")
        success_codes = {"00", "000"}
        if result_code and result_code not in success_codes:
            raise RuntimeError(
                f"API 오류 – code={result_code}, msg={result_msg}"
            )

        items: list[dict[str, Any]] = []
        for item in root.iter("item"):
            row: dict[str, Any] = {}
            for child in item:
                row[child.tag] = child.text.strip() if child.text else ""
            items.append(row)
        return items

    @staticmethod
    def _generate_ym_range(start_ym: str, end_ym: str) -> list[str]:
        """YYYYMM 형식의 시작~종료 범위를 월별 리스트로 반환한다.

        Args:
            start_ym: 시작 연월 (``YYYYMM``).
            end_ym: 종료 연월 (``YYYYMM``).

        Returns:
            ``YYYYMM`` 문자열 리스트.
        """
        months: list[str] = []
        current = pd.Timestamp(f"{start_ym[:4]}-{start_ym[4:]}-01")
        end = pd.Timestamp(f"{end_ym[:4]}-{end_ym[4:]}-01")
        while current <= end:
            months.append(current.strftime("%Y%m"))
            current += pd.DateOffset(months=1)
        return months

    def _fetch_page(
        self,
        endpoint: str,
        lawd_cd: str,
        deal_ymd: str,
        page_no: int = 1,
    ) -> list[dict[str, Any]]:
        """단일 페이지 API 호출을 수행한다.

        Args:
            endpoint: API 엔드포인트 URL.
            lawd_cd: 법정동코드 5자리.
            deal_ymd: 거래연월 ``YYYYMM``.
            page_no: 페이지 번호.

        Returns:
            item 딕셔너리 리스트.
        """
        params = {
            "serviceKey": self.api_key,
            "LAWD_CD": lawd_cd,
            "DEAL_YMD": deal_ymd,
            "pageNo": str(page_no),
            "numOfRows": str(MOLIT_NUM_OF_ROWS),
        }
        resp = requests.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()
        return self._parse_xml_response(resp.text)

    def _fetch_all_pages(
        self,
        endpoint: str,
        lawd_cd: str,
        deal_ymd: str,
    ) -> pd.DataFrame:
        """페이지네이션을 처리하여 전체 데이터를 수집한다.

        numOfRows를 9999로 설정하여 대부분 1회 호출로 처리되지만,
        데이터가 더 있을 경우 다음 페이지를 자동으로 요청한다.

        Args:
            endpoint: API 엔드포인트 URL.
            lawd_cd: 법정동코드 5자리.
            deal_ymd: 거래연월 ``YYYYMM``.

        Returns:
            수집된 전체 거래 데이터 DataFrame.
        """
        all_items: list[dict[str, Any]] = []
        page_no = 1

        while True:
            items = self._fetch_page(endpoint, lawd_cd, deal_ymd, page_no)
            all_items.extend(items)

            # 결과가 numOfRows 미만이면 마지막 페이지
            if len(items) < MOLIT_NUM_OF_ROWS:
                break

            page_no += 1
            time.sleep(MOLIT_API_SLEEP)

        if not all_items:
            return pd.DataFrame()

        return pd.DataFrame(all_items)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def fetch_apt_trade(self, lawd_cd: str, deal_ymd: str) -> pd.DataFrame:
        """특정 지역-월의 아파트 매매 실거래가를 수집한다.

        Args:
            lawd_cd: 법정동코드 5자리.
            deal_ymd: 거래연월 ``YYYYMM``.

        Returns:
            해당 지역-월 매매 거래 DataFrame.
        """
        df = self._fetch_all_pages(MOLIT_APT_TRADE_ENDPOINT, lawd_cd, deal_ymd)
        if df.empty:
            logger.info(f"매매 데이터 없음: {lawd_cd}_{deal_ymd}")
            return df

        out_path = self.save_dir / "apt_trade" / f"{lawd_cd}_{deal_ymd}.parquet"
        df.to_parquet(out_path, index=False)
        logger.debug(f"매매 저장 완료: {out_path.name} ({len(df)}건)")
        return df

    def fetch_apt_rent(self, lawd_cd: str, deal_ymd: str) -> pd.DataFrame:
        """특정 지역-월의 아파트 전월세 실거래가를 수집한다.

        Args:
            lawd_cd: 법정동코드 5자리.
            deal_ymd: 거래연월 ``YYYYMM``.

        Returns:
            해당 지역-월 전월세 거래 DataFrame.
        """
        df = self._fetch_all_pages(MOLIT_APT_RENT_ENDPOINT, lawd_cd, deal_ymd)
        if df.empty:
            logger.info(f"전월세 데이터 없음: {lawd_cd}_{deal_ymd}")
            return df

        out_path = self.save_dir / "apt_rent" / f"{lawd_cd}_{deal_ymd}.parquet"
        df.to_parquet(out_path, index=False)
        logger.debug(f"전월세 저장 완료: {out_path.name} ({len(df)}건)")
        return df

    def run_full_collection(
        self,
        start_ym: str,
        end_ym: str,
        region_codes: dict[str, str] | None = None,
        collect_trade: bool = True,
        collect_rent: bool = True,
    ) -> None:
        """전체 기간 × 전체 지역에 대해 매매/전월세 데이터를 수집한다.

        이미 존재하는 parquet 파일은 건너뛴다 (증분 수집).
        API 호출 실패 시 ``failed_list.json`` 에 기록하고 다음 항목으로 진행한다.
        중간에 rate limit 등으로 연속 실패가 발생하면 대기 시간을 점진적으로
        늘리며 재시도한다.

        Args:
            start_ym: 시작 연월 ``YYYYMM``.
            end_ym: 종료 연월 ``YYYYMM``.
            region_codes: 수집 대상 지역코드 딕셔너리.
                ``None`` 이면 ``ALL_REGIONS`` 사용.
            collect_trade: 매매 데이터 수집 여부.
            collect_rent: 전월세 데이터 수집 여부.
        """
        if region_codes is None:
            region_codes = ALL_REGIONS

        months = self._generate_ym_range(start_ym, end_ym)
        failed: list[dict[str, str]] = []
        consecutive_errors = 0
        max_consecutive_errors = 10

        # 기존 실패 목록 불러오기 (재시도 지원)
        failed_path = self.save_dir / "failed_list.json"

        tasks: list[tuple[str, str, str]] = []
        for lawd_cd, name in region_codes.items():
            for ym in months:
                if collect_trade:
                    tasks.append(("trade", lawd_cd, ym))
                if collect_rent:
                    tasks.append(("rent", lawd_cd, ym))

        logger.info(f"수집 대상: {len(tasks)}건 (지역 {len(region_codes)}개 × 월 {len(months)}개)")

        for task_type, lawd_cd, ym in tqdm(tasks, desc="실거래가 수집"):
            region_name = region_codes.get(lawd_cd, lawd_cd)

            # 증분 수집: 이미 파일이 존재하면 skip
            sub_dir = "apt_trade" if task_type == "trade" else "apt_rent"
            out_path = self.save_dir / sub_dir / f"{lawd_cd}_{ym}.parquet"
            if out_path.exists():
                continue

            try:
                if task_type == "trade":
                    self.fetch_apt_trade(lawd_cd, ym)
                else:
                    self.fetch_apt_rent(lawd_cd, ym)

                consecutive_errors = 0
                time.sleep(MOLIT_API_SLEEP)

            except Exception as exc:
                consecutive_errors += 1
                logger.warning(
                    f"[{task_type}] {region_name}({lawd_cd}) {ym} 실패: {exc}"
                )
                failed.append({
                    "type": task_type,
                    "lawd_cd": lawd_cd,
                    "deal_ymd": ym,
                    "region_name": region_name,
                    "error": str(exc),
                })

                # 연속 실패 시 대기 시간 증가 (rate limit 대비)
                if consecutive_errors >= max_consecutive_errors:
                    wait_time = min(60, consecutive_errors * 5)
                    logger.warning(
                        f"연속 {consecutive_errors}회 실패 – "
                        f"{wait_time}초 대기 후 재개"
                    )
                    time.sleep(wait_time)

                    # 연속 실패가 너무 많으면 중간 저장 후 중단
                    if consecutive_errors >= max_consecutive_errors * 3:
                        logger.error(
                            f"연속 {consecutive_errors}회 실패 – 수집 중단. "
                            f"failed_list.json 확인 후 재실행하세요."
                        )
                        break
                else:
                    time.sleep(MOLIT_API_SLEEP * 2)

        # 실패 목록 저장
        if failed:
            # 기존 실패 목록과 병합
            existing_failed: list[dict[str, str]] = []
            if failed_path.exists():
                existing_failed = json.loads(failed_path.read_text(encoding="utf-8"))

            merged = existing_failed + failed
            failed_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.warning(f"실패 {len(failed)}건 기록됨: {failed_path}")
        else:
            logger.info("모든 수집 완료 (실패 없음)")

    def retry_failed(self) -> None:
        """``failed_list.json`` 에 기록된 실패 항목을 재시도한다."""
        failed_path = self.save_dir / "failed_list.json"
        if not failed_path.exists():
            logger.info("재시도할 실패 목록 없음")
            return

        items: list[dict[str, str]] = json.loads(
            failed_path.read_text(encoding="utf-8")
        )
        still_failed: list[dict[str, str]] = []

        for item in tqdm(items, desc="실패 항목 재시도"):
            task_type = item["type"]
            lawd_cd = item["lawd_cd"]
            ym = item["deal_ymd"]

            try:
                if task_type == "trade":
                    self.fetch_apt_trade(lawd_cd, ym)
                else:
                    self.fetch_apt_rent(lawd_cd, ym)
                time.sleep(MOLIT_API_SLEEP)
            except Exception as exc:
                logger.warning(f"재시도 실패: {lawd_cd}_{ym} – {exc}")
                item["error"] = str(exc)
                still_failed.append(item)
                time.sleep(MOLIT_API_SLEEP * 2)

        if still_failed:
            failed_path.write_text(
                json.dumps(still_failed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.warning(f"재시도 후에도 {len(still_failed)}건 실패")
        else:
            failed_path.unlink(missing_ok=True)
            logger.info("모든 재시도 성공")
