"""국토교통부 실거래가 Raw 데이터 전처리 파이프라인.

수집된 매매/전월세 Raw 데이터를 정제하여 분석에 용이한 형태의
통합 parquet 파일로 변환한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from config.settings import MOLIT_RAW_DIR, PROCESSED_DIR


class DataPreprocessor:
    """국토부 실거래가 데이터를 전처리하는 클래스.

    취소 거래 필터링, 컬럼명 변경, 데이터 타입 변환, 파생 변수 생성 등을 수행한다.
    """

    def __init__(
        self, 
        raw_dir: str | Path | None = None, 
        processed_dir: str | Path | None = None
    ) -> None:
        """DataPreprocessor를 초기화한다.

        Args:
            raw_dir: Raw 데이터 저장 경로.
            processed_dir: 전처리된 데이터 저장 경로.
        """
        self.raw_dir = Path(raw_dir) if raw_dir else MOLIT_RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _load_all_parquets(self, sub_dir: str) -> pd.DataFrame:
        """지정한 하위 디렉토리 내의 모든 parquet 파일을 로드하여 병합한다.

        Args:
            sub_dir: 'apt_trade' 또는 'apt_rent'.

        Returns:
            병합된 DataFrame.
        """
        directory = self.raw_dir / sub_dir
        files = sorted(directory.glob("*.parquet"))
        if not files:
            logger.warning(f"데이터가 없습니다: {directory}")
            return pd.DataFrame()

        dfs = []
        for f in tqdm(files, desc=f"Loading {sub_dir}"):
            try:
                df = pd.read_parquet(f)
                if df.empty:
                    continue
                dfs.append(df)
            except Exception as exc:
                logger.error(f"파일 로드 실패: {f.name} - {exc}")

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _clean_amount(self, series: pd.Series) -> pd.Series:
        """금액 컬럼(문자열)에서 콤마를 제거하고 정수로 변환한다.

        Args:
            series: 변환할 Series.

        Returns:
            정수형 Series.
        """
        return series.astype(str).str.replace(",", "").astype(int)

    # 84㎡ 환산 시 사용하는 기준 면적
    STANDARD_AREA_M2: float = 84.0
    # 1평 = 3.3058㎡ (공식 환산 계수)
    PYEONG_PER_M2: float = 3.3058

    def _create_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매/전월세 공통 기본 컬럼을 생성한다.

        생성되는 컬럼:
            - date: 거래 일자
            - area: 전용면적 (m², 분석용 원본)
            - floor: 층수
            - construction_year: 건축년도
            - age: 건물 경과연수 (거래시점 기준)
            - dong / apt_name: 법정동명 / 아파트명
            - area_repr: 대표 면적 (평형 그룹핑용, 분석에서는 area 사용)
            - dong_repr / apt_name_repr: 분석용 대표 명칭

        Args:
            df: 원본 Raw DataFrame.

        Returns:
            파생 컬럼이 추가된 DataFrame.
        """
        # 일자 생성 (NaN 방지)
        df["dealYear"] = df["dealYear"].astype(str)
        df["dealMonth"] = df["dealMonth"].astype(str).str.zfill(2)
        df["dealDay"] = df["dealDay"].astype(str).str.zfill(2)
        df["date"] = pd.to_datetime(
            df["dealYear"] + "-" + df["dealMonth"] + "-" + df["dealDay"],
            errors="coerce"
        )

        # 수치형 변환
        df["area"] = pd.to_numeric(df["excluUseAr"], errors="coerce")
        df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
        df["construction_year"] = pd.to_numeric(df["buildYear"], errors="coerce").fillna(0).astype(int)

        # 건물 경과연수: 거래시점 연도 - 건축년도
        df["age"] = df["date"].dt.year - df["construction_year"]
        # construction_year 가 0(미상)인 경우 age도 무의미하므로 NaN 처리
        df.loc[df["construction_year"] == 0, "age"] = np.nan

        # 명칭 정제
        df["dong"] = df["umdNm"].str.strip()
        df["apt_name"] = df["aptNm"].str.strip()

        # area_repr: 평형 그룹핑 표시용 (분석에서는 area를 사용할 것)
        df["area_repr"] = np.floor(df["area"]).astype("Int64")
        
        # dong_repr: 법정동명(시군구코드)
        df["dong_repr"] = df["dong"] + "(" + df["sggCd"].astype(str) + ")"
        
        # apt_name_repr: 아파트명(법정동)
        road_nm = df["roadNm"] if "roadNm" in df.columns else ""
        df["apt_name_repr"] = df["apt_name"] + "-" + road_nm + "(" + df["dong"] + ")"

        return df

    def _add_trade_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매 데이터에 가격 파생 컬럼을 추가한다.

        추가되는 컬럼:
            - price_per_m2: ㎡당 가격 (만원/m²) — 1차 탐색·이상치 탐지용
            - price_per_py: 평당 가격 (만원/평) — 직관적 비교용
            - price_std84: 84㎡ 환산 가격 (만원) — 서로 다른 평형 비교용 (선형 환산)

        Note:
            price_std84는 단순 선형 비례(price × 84/area) 환산이다.
            실제로는 면적 탄력성(β<1)이 존재하므로, 보다 정밀한 비교가
            필요하면 단지별/법정동별 β를 추정하여 보정해야 한다.
            (price_std84_beta = price × (84/area)^β)

        Args:
            df: price, area 컬럼이 있는 매매 DataFrame.

        Returns:
            가격 파생 컬럼이 추가된 DataFrame.
        """
        safe_area = df["area"].replace(0, np.nan)

        # ㎡당 가격
        df["price_per_m2"] = (df["price"] / safe_area).round(2)
        # 평당 가격
        df["price_per_py"] = (df["price"] / (safe_area / self.PYEONG_PER_M2)).round(2)
        # 84㎡ 환산 가격 (선형 비례)
        df["price_std84"] = (df["price"] * (self.STANDARD_AREA_M2 / safe_area)).round(0).astype("Int64")

        return df

    def _add_rent_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """전월세 데이터에 가격 파생 컬럼을 추가한다.

        추가되는 컬럼:
            - deposit_per_m2: 보증금 ㎡당 가격 (만원/m²)
            - deposit_per_py: 보증금 평당 가격 (만원/평)
            - deposit_std84: 보증금 84㎡ 환산 (만원)
            - monthly_rent_per_m2: 월세 ㎡당 가격 (만원/m²)
            - monthly_rent_per_py: 월세 평당 가격 (만원/평)

        Args:
            df: deposit, monthly_rent, area 컬럼이 있는 전월세 DataFrame.

        Returns:
            가격 파생 컬럼이 추가된 DataFrame.
        """
        safe_area = df["area"].replace(0, np.nan)

        # 보증금 단가
        df["deposit_per_m2"] = (df["deposit"] / safe_area).round(2)
        df["deposit_per_py"] = (df["deposit"] / (safe_area / self.PYEONG_PER_M2)).round(2)
        # 보증금 84㎡ 환산
        df["deposit_std84"] = (df["deposit"] * (self.STANDARD_AREA_M2 / safe_area)).round(0).astype("Int64")

        # 월세 단가 (월세가 0인 전세 거래도 포함)
        df["monthly_rent_per_m2"] = (df["monthly_rent"] / safe_area).round(2)
        df["monthly_rent_per_py"] = (df["monthly_rent"] / (safe_area / self.PYEONG_PER_M2)).round(2)

        return df

    def _save_in_chunks(self, df: pd.DataFrame, prefix: str) -> None:
        """데이터를 1년 단위로 쪼개서 저장한다.

        Args:
            df: 저장할 DataFrame (date 컬럼 필수).
            prefix: 파일명 접두사 (예: 'apt_trade', 'apt_rent').
        """
        if df.empty:
            return

        years = df["date"].dt.year.unique()
        for year in sorted(years):
            if pd.isna(year):
                continue
            
            year_val = int(year)
            chunk = df[df["date"].dt.year == year_val].copy()
            
            if chunk.empty:
                continue
                
            out_path = self.processed_dir / f"{prefix}_{year_val}.parquet"
            chunk.to_parquet(out_path, index=False)
            logger.info(f"조각 저장 완료: {out_path.name} ({len(chunk)}건)")

    def preprocess_trade(self) -> pd.DataFrame:
        """매매 데이터를 전처리한다.

        취소된 거래(cdealType == 'O')를 제외하고, 
        schema에 정의된 컬럼만 추출하여 1년 단위 조각으로 저장한다.
        """
        logger.info("매매 데이터 전처리 시작")
        df = self._load_all_parquets("apt_trade")
        if df.empty:
            return df

        # 1) 취소 거래 제외
        if "cdealType" in df.columns:
            initial_count = len(df)
            df = df[df["cdealType"] != "O"].copy()
            cancelled_count = initial_count - len(df)
            logger.info(f"취소 거래 {cancelled_count}건 제외 완료")

        # 2) 기본 컬럼 생성
        df = self._create_base_columns(df)

        # 3) 매매 전용 컬럼: price (dealAmount)
        df["price"] = self._clean_amount(df["dealAmount"])

        # 4) 가격 파생 컬럼 생성
        df = self._add_trade_price_columns(df)

        # 5) 필요한 컬럼만 선택
        cols = [
            "date", "price", "price_per_m2", "price_per_py", "price_std84",
            "area", "floor", "construction_year", "age",
            "dong", "apt_name", "area_repr", "dong_repr", "apt_name_repr", "aptSeq"
        ]
        cols = [c for c in cols if c in df.columns]
        processed_df = df[cols].sort_values("date").reset_index(drop=True)

        # 6) 조각내어 저장 (GitHub 용량 제한 회피)
        self._save_in_chunks(processed_df, "apt_trade")
        
        # 하위 호환성 또는 집계 편의를 위해 전체 파일도 일단 유지 (필요 없으면 삭제 가능)
        # git push 시에는 .gitignore 에 등록하거나 삭제하는 것이 좋음
        out_path = self.processed_dir / "apt_trade.parquet"
        processed_df.to_parquet(out_path, index=False)
        logger.info(f"매매 전처리 완료 (전체): {out_path} ({len(processed_df)}건)")

        return processed_df

    def preprocess_rent(self) -> pd.DataFrame:
        """전월세 데이터를 전처리하여 1년 단위 조각으로 저장한다."""
        logger.info("전월세 데이터 전처리 시작")
        df = self._load_all_parquets("apt_rent")
        if df.empty:
            return df

        if "cdealType" in df.columns:
            df = df[df["cdealType"] != "O"].copy()

        # 1) 기본 컬럼 생성
        df = self._create_base_columns(df)

        # 2) 전월세 전용 컬럼
        df["deposit"] = self._clean_amount(df["deposit"])
        df["monthly_rent"] = self._clean_amount(df["monthlyRent"])

        # rentType 생성 (Raw 데이터에 없는 경우 대비: 월세가 0이면 전세, 아니면 월세)
        if "rentType" not in df.columns:
            df["rentType"] = np.where(df["monthly_rent"] == 0, "전세", "월세")
        else:
            # 기존 rentType이 있다면 공백 제거 및 표준화
            df["rentType"] = df["rentType"].astype(str).str.strip()
            # "준월세", "준전세" 등도 "월세"로 간주하거나 그대로 둘 수 있지만, 
            # 여기서는 대시보드 표시를 위해 "전세" / "월세"로 단순화 시도 (필요 시 수정)
            df.loc[df["rentType"].isin(["월세", "준월세", "준전세"]), "rentType"] = "월세"
            df.loc[df["rentType"] == "전세", "rentType"] = "전세"

        # 3) 가격 파생 컬럼 생성
        df = self._add_rent_price_columns(df)

        # 4) 기타 컬럼
        rename_map = {
            "contractType": "contract_type",
            "contractTerm": "contract_term",
            "useRRRight": "use_rr_right",
            "preDeposit": "pre_deposit",
            "preMonthlyRent": "pre_monthly_rent"
        }
        for old_col, new_col in rename_map.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]

        # 5) 필요한 컬럼 선택
        cols = [
            "date", "deposit", "deposit_per_m2", "deposit_per_py", "deposit_std84",
            "monthly_rent", "monthly_rent_per_m2", "monthly_rent_per_py",
            "area", "floor", "construction_year", "age",
            "dong", "apt_name", "area_repr", "dong_repr", "apt_name_repr", "aptSeq",
            "rentType", "contract_type", "contract_term", "use_rr_right"
        ]
        cols = [c for c in cols if c in df.columns]
        processed_df = df[cols].sort_values("date").reset_index(drop=True)

        # 6) 조각내어 저장
        self._save_in_chunks(processed_df, "apt_rent")

        out_path = self.processed_dir / "apt_rent.parquet"
        processed_df.to_parquet(out_path, index=False)
        logger.info(f"전월세 전처리 완료 (전체): {out_path} ({len(processed_df)}건)")

        return processed_df


if __name__ == "__main__":
    # 간단한 실행 테스트
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_trade()
    preprocessor.preprocess_rent()
