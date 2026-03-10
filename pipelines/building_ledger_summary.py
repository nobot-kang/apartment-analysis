"""건축물대장(표제부) 데이터를 요약하여 아파트 단지 정보 lookup table을 생성하는 모듈.

국토교통부 건축물대장 Raw 데이터에서 단지별 고유 정보를 추출하고, 
거래 데이터와 결합 가능한 'apartment_info.parquet' 파일을 생성한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from config.settings import MOLIT_RAW_DIR, PROCESSED_DIR


class BuildingLedgerSummarizer:
    """건축물대장 데이터를 요약하는 클래스.

    단지별(aptSeq)로 가장 적절한 대장 정보를 선택하고, 
    분석에 필요한 주요 컬럼을 추출한다.
    """

    def __init__(
        self, 
        raw_dir: str | Path | None = None, 
        processed_dir: str | Path | None = None
    ) -> None:
        """BuildingLedgerSummarizer를 초기화한다.

        Args:
            raw_dir: Raw 데이터 저장 경로.
            processed_dir: 전처리된 데이터 저장 경로.
        """
        self.raw_dir = Path(raw_dir) if raw_dir else MOLIT_RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_path = self.raw_dir / "building_ledger" / "building_ledger_all.parquet"
        self.out_path = self.processed_dir / "apartment_info.parquet"

    def _clean_date(self, series: pd.Series) -> pd.Series:
        """사용승인일(YYYYMMDD) 등 날짜 데이터를 정제한다.

        - 8자리가 아닐 경우 NaT 처리
        - datetime64[ns] 타입으로 변환
        """
        # 8자리 숫자가 아닌 경우 등 비정상 데이터 처리
        s = series.astype(str).str.replace(".0", "", regex=False).str.strip()
        # "0", "", "None" 등 처리
        s = s.replace(["0", "", "None", "nan", "NaN"], np.nan)
        
        # 8자리 미만인 경우(예: 199901)는 뒤를 01로 채움 (선택 사항)
        # 여기서는 8자리인 것만 우선 살림
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    def summarize(self) -> pd.DataFrame:
        """건축물대장 데이터를 요약하여 단지 정보 lookup table을 생성한다."""
        if not self.raw_path.exists():
            logger.error(f"원본 파일이 없습니다: {self.raw_path}")
            return pd.DataFrame()

        logger.info(f"건축물대장 로드 중: {self.raw_path}")
        df = pd.read_parquet(self.raw_path)
        
        if df.empty:
            logger.warning("데이터가 비어 있습니다.")
            return df

        # 1) 단지 구분용 aptSeq 확인 (없으면 제외)
        if "aptSeq" not in df.columns:
            logger.error("데이터에 aptSeq 컬럼이 없습니다.")
            return pd.DataFrame()
        
        df = df.dropna(subset=["aptSeq"]).copy()
        logger.info(f"초기 로드: {len(df)}건")

        # 2) 우선순위 선정 (regstrGbCd == "2" 인 집합건축물 우선)
        # regstrGbCd가 "2"인 행이 앞으로 오도록 정렬 후 aptSeq 기준 중복 제거
        # regstrGbCd: 1(총괄), 2(집합), 3(일반)
        # "2"를 0순위, 나머지를 후순위로 취급하는 logic
        df["_priority"] = df["regstrGbCd"].apply(lambda x: 0 if str(x) == "2" else 1)
        df = df.sort_values(by=["aptSeq", "_priority", "crtnDay"], ascending=[True, True, False])
        
        # 중복 제거 (aptSeq당 1개만 남김)
        summary_df = df.drop_duplicates(subset=["aptSeq"], keep="first").copy()
        logger.info(f"중복 제거 후: {len(summary_df)}건")

        # 3) 주요 컬럼 추출 및 정제
        # - bldNm: 건물명 (단지명)
        # - useAprDay: 사용승인일 (준공일)
        # - platArea: 대지면적
        # - vlRatEstmTotArea: 용적률산정연면적
        # - hhldCnt: 세대수
        # - totArea: 연면적
        # - vlRat: 용적률
        # - bcRat: 건폐율
        # - grndFlrCnt: 지상층수
        # - ugrndFlrCnt: 지하층수
        
        cols_map = {
            "aptSeq": "aptSeq",
            "bldNm": "apt_name_ledger",
            "useAprDay": "completion_date",
            "platArea": "land_area",
            "vlRatEstmTotArea": "floor_area_ratio_total_area",
            "totArea": "total_area",
            "vlRat": "floor_area_ratio",
            "bcRat": "building_coverage_ratio",
            "hhldCnt": "household_count",
            "grndFlrCnt": "ground_floor_count",
            "ugrndFlrCnt": "underground_floor_count",
            "platPlc": "address",
            "sigunguCd": "sigungu_code",
            "bjdongCd": "bjdong_code"
        }
        
        # 존재하는 컬럼만 선택
        available_cols = [c for c in cols_map.keys() if c in summary_df.columns]
        summary_df = summary_df[available_cols].rename(columns={c: cols_map[c] for c in available_cols})

        # 수치형 변환
        numeric_cols = [
            "land_area", "floor_area_ratio_total_area", "total_area", 
            "floor_area_ratio", "building_coverage_ratio", "household_count",
            "ground_floor_count", "underground_floor_count"
        ]
        for col in numeric_cols:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0)

        # 날짜 정제
        if "completion_date" in summary_df.columns:
            summary_df["completion_date"] = self._clean_date(summary_df["completion_date"])

        # 4) 저장
        summary_df = summary_df.sort_values("aptSeq").reset_index(drop=True)
        summary_df.to_parquet(self.out_path, index=False)
        logger.info(f"아파트 정보 lookup table 생성 완료: {self.out_path} ({len(summary_df)}건)")

        return summary_df


if __name__ == "__main__":
    summarizer = BuildingLedgerSummarizer()
    summarizer.summarize()
