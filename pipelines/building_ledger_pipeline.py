"""Building-ledger collection pipeline for apartment complexes."""

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
    BUILDING_LEDGER_RECAP_ENDPOINT,
    BUILDING_LEDGER_TITLE_ENDPOINT,
    MOLIT_API_SLEEP,
    MOLIT_RAW_DIR,
    get_api_key,
)
from pipelines.apartment_list import ApartmentListManager

SERVICE_DEFS: dict[str, dict[str, str]] = {
    "title": {
        "endpoint": BUILDING_LEDGER_TITLE_ENDPOINT,
        "all_file": "building_ledger_all.parquet",
        "failed_file": "failed_list_building.json",
        "service_label": "표제부",
        "expected_kind": "표제부",
    },
    "recap": {
        "endpoint": BUILDING_LEDGER_RECAP_ENDPOINT,
        "all_file": "building_ledger_recap_all.parquet",
        "failed_file": "failed_list_building_recap.json",
        "service_label": "총괄표제부",
        "expected_kind": "총괄",
    },
}


class BuildingLedgerPipeline:
    """Collect title and recap title building-ledger records by ``aptSeq``."""

    def __init__(
        self,
        api_key: str | None = None,
        save_dir: str | Path | None = None,
        molit_dir: str | Path | None = None,
    ) -> None:
        self.api_key = api_key or get_api_key("MOLIT_API_KEY")
        self.save_dir = Path(save_dir) if save_dir else MOLIT_RAW_DIR / "building_ledger"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.apt_manager = ApartmentListManager(molit_dir=Path(molit_dir) if molit_dir else MOLIT_RAW_DIR)

    def _service_path(self, service: str) -> Path:
        return self.save_dir / SERVICE_DEFS[service]["all_file"]

    def _failed_path(self, service: str) -> Path:
        return self.save_dir / SERVICE_DEFS[service]["failed_file"]

    def _fetch_single(
        self,
        *,
        endpoint: str,
        sigungu_cd: str,
        bjdong_cd: str,
        bun: str,
        ji: str,
    ) -> pd.DataFrame:
        params = {
            "serviceKey": self.api_key,
            "sigunguCd": sigungu_cd,
            "bjdongCd": bjdong_cd,
            "bun": str(bun).zfill(4),
            "ji": str(ji).zfill(4),
            "numOfRows": 100,
            "pageNo": 1,
        }
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        return self._parse_response(response.text)

    @staticmethod
    def _parse_response(xml_text: str) -> pd.DataFrame:
        data_dict = xmltodict.parse(xml_text)
        response = data_dict.get("response", {})
        header = response.get("header", {})
        if header.get("resultCode") not in ("00", "000"):
            return pd.DataFrame()

        items = response.get("body", {}).get("items")
        if not items or items.get("item") is None:
            return pd.DataFrame()

        item_list = items["item"]
        if isinstance(item_list, dict):
            item_list = [item_list]

        df = pd.DataFrame(item_list)
        if "regstrGbCd" in df.columns:
            filtered = df[df["regstrGbCd"].astype(str) == "2"]
            if not filtered.empty:
                return filtered
        return df

    def fetch_for_apartment(self, apt_row: pd.Series, service: str = "title") -> pd.DataFrame:
        if service not in SERVICE_DEFS:
            raise ValueError(f"Unsupported building-ledger service: {service}")

        service_def = SERVICE_DEFS[service]
        df = self._fetch_single(
            endpoint=service_def["endpoint"],
            sigungu_cd=str(apt_row["sggCd"]),
            bjdong_cd=str(apt_row["umdCd"]),
            bun=str(apt_row["bonbun"]),
            ji=str(apt_row["bubun"]),
        )
        if not df.empty:
            df["aptSeq"] = apt_row["aptSeq"]
            df["ledger_api_type"] = service
        return df

    def _load_collected_seqs(self, service: str) -> set[str]:
        path = self._service_path(service)
        if not path.exists():
            return set()
        try:
            existing = pd.read_parquet(path, columns=["aptSeq"])
            return {str(value) for value in existing["aptSeq"].dropna().unique()}
        except Exception:
            return set()

    def _save_batch(self, batch: list[pd.DataFrame], service: str) -> None:
        if not batch:
            return

        combined = pd.concat(batch, ignore_index=True)
        target_path = self._service_path(service)
        if target_path.exists():
            existing = pd.read_parquet(target_path)
            combined = pd.concat([existing, combined], ignore_index=True)
        combined = combined.drop_duplicates(subset=["aptSeq"], keep="last")
        combined.to_parquet(target_path, index=False)
        logger.info(
            "{} batch saved: {} rows -> {}",
            SERVICE_DEFS[service]["service_label"],
            len(combined),
            target_path.name,
        )

    def _persist_failed_items(self, service: str, failed_items: list[dict[str, str]]) -> None:
        if not failed_items:
            logger.info("{} collection finished with no failed items", SERVICE_DEFS[service]["service_label"])
            return

        failed_path = self._failed_path(service)
        existing_failed: list[dict[str, str]] = []
        if failed_path.exists():
            existing_failed = json.loads(failed_path.read_text(encoding="utf-8"))
        merged = existing_failed + failed_items
        failed_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.warning(
            "{} collection recorded {} failed items at {}",
            SERVICE_DEFS[service]["service_label"],
            len(failed_items),
            failed_path,
        )

    def run_collection(self, service: str, force_rebuild_list: bool = False) -> None:
        if service not in SERVICE_DEFS:
            raise ValueError(f"Unsupported building-ledger service: {service}")

        if force_rebuild_list or not self.apt_manager.list_path.exists():
            self.apt_manager.build_list()

        apt_params = self.apt_manager.get_building_ledger_params()
        if apt_params.empty:
            logger.warning("No apartments are available for building-ledger collection.")
            return

        collected_seqs = self._load_collected_seqs(service)
        pending = apt_params[~apt_params["aptSeq"].astype(str).isin(collected_seqs)].copy()
        logger.info(
            "{} collection pending: {} / {} apartments",
            SERVICE_DEFS[service]["service_label"],
            len(pending),
            len(apt_params),
        )
        if pending.empty:
            logger.info("{} collection already up to date", SERVICE_DEFS[service]["service_label"])
            return

        failed: list[dict[str, str]] = []
        batch: list[pd.DataFrame] = []
        consecutive_errors = 0
        batch_size = 100

        for _, row in tqdm(pending.iterrows(), total=len(pending), desc=f"{SERVICE_DEFS[service]['service_label']} 수집"):
            apt_seq = str(row["aptSeq"])
            try:
                df = self.fetch_for_apartment(row, service=service)
                if not df.empty:
                    batch.append(df)
                if len(batch) >= batch_size:
                    self._save_batch(batch, service)
                    batch = []
                consecutive_errors = 0
                time.sleep(MOLIT_API_SLEEP)
            except Exception as exc:
                consecutive_errors += 1
                logger.warning("{} collection failed for {}: {}", SERVICE_DEFS[service]["service_label"], apt_seq, exc)
                failed.append(
                    {
                        "service": service,
                        "aptSeq": apt_seq,
                        "sggCd": str(row["sggCd"]),
                        "umdCd": str(row["umdCd"]),
                        "bonbun": str(row["bonbun"]),
                        "bubun": str(row["bubun"]),
                        "error": str(exc),
                    }
                )
                if consecutive_errors >= 10:
                    wait_seconds = min(60, consecutive_errors * 5)
                    logger.warning(
                        "{} collection hit {} consecutive errors; sleeping {}s",
                        SERVICE_DEFS[service]["service_label"],
                        consecutive_errors,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    if consecutive_errors >= 30:
                        logger.error("{} collection aborted after repeated failures", SERVICE_DEFS[service]["service_label"])
                        break
                else:
                    time.sleep(MOLIT_API_SLEEP * 2)

        if batch:
            self._save_batch(batch, service)
        self._persist_failed_items(service, failed)

    def run_full_collection(
        self,
        force_rebuild_list: bool = False,
        *,
        collect_title: bool = True,
        collect_recap: bool = True,
    ) -> None:
        if collect_title:
            self.run_collection("title", force_rebuild_list=force_rebuild_list)
            force_rebuild_list = False
        if collect_recap:
            self.run_collection("recap", force_rebuild_list=force_rebuild_list)

    def retry_failed(self, service: str | None = None) -> None:
        services = [service] if service else list(SERVICE_DEFS.keys())
        for service_name in services:
            if service_name not in SERVICE_DEFS:
                raise ValueError(f"Unsupported building-ledger service: {service_name}")

            failed_path = self._failed_path(service_name)
            if not failed_path.exists():
                logger.info("No failed-item file found for {}", SERVICE_DEFS[service_name]["service_label"])
                continue

            items: list[dict[str, str]] = json.loads(failed_path.read_text(encoding="utf-8"))
            still_failed: list[dict[str, str]] = []
            batch: list[pd.DataFrame] = []
            for item in tqdm(items, desc=f"{SERVICE_DEFS[service_name]['service_label']} 재시도"):
                try:
                    row = pd.Series(item)
                    df = self.fetch_for_apartment(row, service=service_name)
                    if not df.empty:
                        batch.append(df)
                    time.sleep(MOLIT_API_SLEEP)
                except Exception as exc:
                    item["error"] = str(exc)
                    still_failed.append(item)
                    logger.warning("{} retry failed for {}: {}", SERVICE_DEFS[service_name]["service_label"], item["aptSeq"], exc)
                    time.sleep(MOLIT_API_SLEEP * 2)

            if batch:
                self._save_batch(batch, service_name)

            if still_failed:
                failed_path.write_text(json.dumps(still_failed, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.warning(
                    "{} retry still has {} failed items",
                    SERVICE_DEFS[service_name]["service_label"],
                    len(still_failed),
                )
            else:
                failed_path.unlink(missing_ok=True)
                logger.info("{} retry completed successfully", SERVICE_DEFS[service_name]["service_label"])
