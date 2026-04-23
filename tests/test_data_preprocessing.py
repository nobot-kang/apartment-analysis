from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines.data_preprocessing import DataPreprocessor


def _make_raw_row(
    *,
    dealYear: int = 2021,
    dealMonth: int = 1,
    dealDay: int = 3,
    excluUseAr: float = 84.91,
    floor: int = 10,
    buildYear: int = 2010,
    umdNm: str = "인의동",
    aptNm: str = "테스트아파트",
    sggCd: str = "11110",
    roadNm: str = "테스트로",
    dealAmount: str = "100,000",
    aptSeq: str = "11110-1",
    cdealType: str = "",
    dealingGbn: str = "중개거래",
) -> dict:
    return dict(
        dealYear=dealYear,
        dealMonth=dealMonth,
        dealDay=dealDay,
        excluUseAr=excluUseAr,
        floor=floor,
        buildYear=buildYear,
        umdNm=umdNm,
        aptNm=aptNm,
        sggCd=sggCd,
        roadNm=roadNm,
        dealAmount=dealAmount,
        aptSeq=aptSeq,
        cdealType=cdealType,
        dealingGbn=dealingGbn,
    )


def test_preprocess_trade_excludes_direct_and_cancelled_and_writes_summary(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    trade_dir = raw_dir / "apt_trade"
    trade_dir.mkdir(parents=True)
    processed_dir = tmp_path / "processed"

    raw_df = pd.DataFrame(
        [
            {
                "dealYear": 2021,
                "dealMonth": 1,
                "dealDay": 3,
                "excluUseAr": 84.91,
                "floor": 10,
                "buildYear": 2010,
                "umdNm": "인의동",
                "aptNm": "테스트아파트",
                "sggCd": "11110",
                "roadNm": "테스트로",
                "dealAmount": "100,000",
                "aptSeq": "11110-1",
                "cdealType": "",
                "dealingGbn": "중개거래",
            },
            {
                "dealYear": 2021,
                "dealMonth": 1,
                "dealDay": 10,
                "excluUseAr": 84.91,
                "floor": 11,
                "buildYear": 2010,
                "umdNm": "인의동",
                "aptNm": "테스트아파트",
                "sggCd": "11110",
                "roadNm": "테스트로",
                "dealAmount": "110,000",
                "aptSeq": "11110-1",
                "cdealType": "",
                "dealingGbn": "직거래",
            },
            {
                "dealYear": 2021,
                "dealMonth": 1,
                "dealDay": 15,
                "excluUseAr": 84.91,
                "floor": 12,
                "buildYear": 2010,
                "umdNm": "인의동",
                "aptNm": "테스트아파트",
                "sggCd": "11110",
                "roadNm": "테스트로",
                "dealAmount": "120,000",
                "aptSeq": "11110-1",
                "cdealType": "O",
                "dealingGbn": "중개거래",
            },
        ]
    )
    raw_df.to_parquet(trade_dir / "sample.parquet", index=False)

    preprocessor = DataPreprocessor(raw_dir=raw_dir, processed_dir=processed_dir)
    processed_df = preprocessor.preprocess_trade()

    assert len(processed_df) == 1
    assert processed_df.iloc[0]["price"] == 100000
    assert processed_df.iloc[0]["aptSeq"] == "11110-1"

    summary_path = tmp_path / "preprocessed_plus" / "trade_filter_yearly_summary.parquet"
    assert summary_path.exists()

    summary_df = pd.read_parquet(summary_path)
    row = summary_df[(summary_df["sggCd"] == "11110") & (summary_df["year"] == 2021)].iloc[0]
    assert int(row["total_trade_count"]) == 3
    assert int(row["cancel_trade_count"]) == 1
    assert int(row["direct_trade_count"]) == 1
    assert round(float(row["cancel_ratio_pct"]), 2) == 33.33
    assert round(float(row["direct_ratio_pct"]), 2) == 33.33


def test_trade_filter_summary_overlap_decomposition(tmp_path: Path) -> None:
    """cancel/direct overlap 4종 분해 및 귀속 규칙 검증 (테스트 #11)."""
    raw_dir = tmp_path / "raw"
    trade_dir = raw_dir / "apt_trade"
    trade_dir.mkdir(parents=True)
    processed_dir = tmp_path / "processed"

    raw_df = pd.DataFrame(
        [
            # cancel only: cdealType=O, dealingGbn=중개거래
            _make_raw_row(cdealType="O", dealingGbn="중개거래", dealDay=1),
            # direct only: cdealType="", dealingGbn=직거래
            _make_raw_row(cdealType="", dealingGbn="직거래", dealDay=2),
            # cancel + direct: cdealType=O AND dealingGbn=직거래
            _make_raw_row(cdealType="O", dealingGbn="직거래", dealDay=3),
            # neither: 정상 거래
            _make_raw_row(cdealType="", dealingGbn="중개거래", dealDay=4),
        ]
    )
    raw_df.to_parquet(trade_dir / "sample.parquet", index=False)

    preprocessor = DataPreprocessor(raw_dir=raw_dir, processed_dir=processed_dir)
    preprocessor.preprocess_trade()

    summary_path = tmp_path / "preprocessed_plus" / "trade_filter_yearly_summary.parquet"
    summary_df = pd.read_parquet(summary_path)
    row = summary_df[(summary_df["sggCd"] == "11110") & (summary_df["year"] == 2021)].iloc[0]

    # 신규 4개 컬럼 값 검증
    assert int(row["cancel_only_count"]) == 1, "cancel only"
    assert int(row["direct_only_count"]) == 1, "direct only"
    assert int(row["cancel_and_direct_count"]) == 1, "cancel and direct"
    assert int(row["after_cancel_direct_count"]) == 1, "neither"

    # raw 항등식: cancel_only + direct_only + cancel_and_direct + after = total
    assert (
        int(row["cancel_only_count"])
        + int(row["direct_only_count"])
        + int(row["cancel_and_direct_count"])
        + int(row["after_cancel_direct_count"])
    ) == int(row["total_trade_count"])

    # legacy 하위호환: cancel_trade_count = cancel_only + cancel_and_direct
    assert int(row["cancel_trade_count"]) == int(row["cancel_only_count"]) + int(row["cancel_and_direct_count"])
    # legacy 하위호환: direct_trade_count = direct_only + cancel_and_direct
    assert int(row["direct_trade_count"]) == int(row["direct_only_count"]) + int(row["cancel_and_direct_count"])

    # funnel 귀속 규칙 (변경 6 잠금): removed_cancel = cancel_only + cancel_and_direct, removed_direct = direct_only
    removed_cancel = int(row["cancel_only_count"]) + int(row["cancel_and_direct_count"])
    removed_direct = int(row["direct_only_count"])
    assert removed_cancel == 2  # 1 cancel_only + 1 cancel_and_direct
    assert removed_direct == 1  # 1 direct_only

    # 정합성 게이트 불변식: preprocess_trade() 후 apt_trade_*.parquet 행수 == after_cancel_direct_count
    apt_trade_files = list(processed_dir.glob("apt_trade_[0-9][0-9][0-9][0-9].parquet"))
    if apt_trade_files:
        actual_rows = sum(len(pd.read_parquet(f)) for f in apt_trade_files)
        expected_rows = int(
            summary_df[summary_df["sggCd"].str.match(r"^\d{5}$")]["after_cancel_direct_count"].sum()
        )
        assert actual_rows == expected_rows
