"""아파트 고유 목록 관리 모듈.

수집된 매매/전월세 Raw 데이터에서 중복 없는 아파트 목록을 추출·관리한다.
이 목록은 건축물대장 등 추가 데이터 수집의 기준 키로 활용된다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import MOLIT_RAW_DIR


# 아파트 목록에 보존할 컬럼 (trade 데이터 기준)
_TRADE_KEY_COLS: list[str] = [
    "aptSeq",
    "aptNm",
    "sggCd",
    "umdCd",
    "umdNm",
    "bonbun",
    "bubun",
    "jibun",
    "buildYear",
    "roadNm",
    "roadNmBonbun",
    "roadNmBubun",
    "roadNmCd",
    "roadNmSggCd",
]

# rent 데이터에서 보충할 수 있는 컬럼
_RENT_KEY_COLS: list[str] = [
    "aptSeq",
    "aptNm",
    "sggCd",
    "umdNm",
    "jibun",
    "buildYear",
]


class ApartmentListManager:
    """수집된 Raw 데이터에서 고유 아파트 목록을 관리하는 클래스.

    ``aptSeq`` 를 기본 키로 사용하며, 건축물대장 API 호출에 필요한
    ``sggCd``, ``umdCd``, ``bonbun``, ``bubun`` 등의 정보를 함께 보관한다.

    Attributes:
        molit_dir: 국토부 Raw 데이터 디렉토리.
        list_path: 아파트 목록 파일 경로.
    """

    def __init__(self, molit_dir: str | Path | None = None) -> None:
        """ApartmentListManager를 초기화한다.

        Args:
            molit_dir: 국토부 Raw 데이터 경로. 기본값은 ``data/raw/molit``.
        """
        self.molit_dir = Path(molit_dir) if molit_dir else MOLIT_RAW_DIR
        self.list_path = self.molit_dir / "apartment_list.parquet"

    def _scan_parquet_dir(
        self,
        directory: Path,
        key_cols: list[str],
    ) -> pd.DataFrame:
        """디렉토리 내 parquet 파일들에서 지정 컬럼을 추출한다.

        Args:
            directory: parquet 파일이 있는 디렉토리.
            key_cols: 추출할 컬럼 리스트.

        Returns:
            추출된 DataFrame (중복 포함).
        """
        files = sorted(directory.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        dfs: list[pd.DataFrame] = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                available = [c for c in key_cols if c in df.columns]
                if "aptSeq" not in available:
                    continue
                dfs.append(df[available])
            except Exception as exc:
                logger.warning(f"파일 로드 실패: {f.name} – {exc}")

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def build_list(self) -> pd.DataFrame:
        """매매/전월세 데이터를 스캔하여 고유 아파트 목록을 생성·저장한다.

        trade 데이터를 우선 사용하고, rent에만 존재하는 아파트를 보충한다.
        ``aptSeq`` 기준으로 중복을 제거하며, trade 데이터의 행을 우선한다.

        Returns:
            고유 아파트 목록 DataFrame.
        """
        logger.info("아파트 목록 생성 시작")

        # 1) trade 데이터에서 추출 (bonbun, bubun, umdCd 포함)
        trade_dir = self.molit_dir / "apt_trade"
        trade_df = self._scan_parquet_dir(trade_dir, _TRADE_KEY_COLS)

        # 2) rent 데이터에서 추출 (보충용)
        rent_dir = self.molit_dir / "apt_rent"
        rent_df = self._scan_parquet_dir(rent_dir, _RENT_KEY_COLS)

        # 3) trade 우선 deduplicate
        if not trade_df.empty:
            trade_unique = trade_df.drop_duplicates(subset=["aptSeq"], keep="first")
        else:
            trade_unique = pd.DataFrame()

        if not rent_df.empty:
            rent_unique = rent_df.drop_duplicates(subset=["aptSeq"], keep="first")
        else:
            rent_unique = pd.DataFrame()

        # trade에 없는 rent-only 아파트 보충
        if not trade_unique.empty and not rent_unique.empty:
            trade_seqs = set(trade_unique["aptSeq"])
            rent_only = rent_unique[~rent_unique["aptSeq"].isin(trade_seqs)]
            combined = pd.concat([trade_unique, rent_only], ignore_index=True)
        elif not trade_unique.empty:
            combined = trade_unique
        elif not rent_unique.empty:
            combined = rent_unique
        else:
            logger.warning("수집된 매매/전월세 데이터가 없습니다.")
            return pd.DataFrame()

        combined = combined.sort_values("aptSeq").reset_index(drop=True)

        # 4) 저장
        combined.to_parquet(self.list_path, index=False)
        logger.info(
            f"아파트 목록 저장 완료: {self.list_path.name} "
            f"({len(combined)}건, trade={len(trade_unique)}, rent-only={len(combined) - len(trade_unique)})"
        )

        return combined

    def load_list(self) -> pd.DataFrame:
        """저장된 아파트 목록을 로드한다. 없으면 새로 생성한다.

        Returns:
            아파트 목록 DataFrame.
        """
        if self.list_path.exists():
            df = pd.read_parquet(self.list_path)
            logger.info(f"기존 아파트 목록 로드: {len(df)}건")
            return df
        return self.build_list()

    def get_building_ledger_params(self) -> pd.DataFrame:
        """건축물대장 API 호출에 필요한 파라미터 DataFrame을 반환한다.

        ``sggCd``, ``umdCd``, ``bonbun``, ``bubun`` 이 모두 있는 행만 반환한다.
        (이 정보가 없는 rent-only 아파트는 제외된다.)

        Returns:
            건축물대장 조회 가능한 아파트 목록 DataFrame.
        """
        df = self.load_list()
        if df.empty:
            return df

        required = ["aptSeq", "sggCd", "umdCd", "bonbun", "bubun"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"필수 컬럼 누락: {missing}")
            return pd.DataFrame()

        valid = df.dropna(subset=required)
        # bonbun/bubun이 빈 문자열인 행 제외
        valid = valid[
            (valid["bonbun"].astype(str).str.strip() != "")
            & (valid["sggCd"].astype(str).str.strip() != "")
            & (valid["umdCd"].astype(str).str.strip() != "")
        ]
        logger.info(f"건축물대장 조회 가능 아파트: {len(valid)}건 / 전체 {len(df)}건")
        return valid.reset_index(drop=True)
