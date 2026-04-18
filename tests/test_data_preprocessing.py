from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines.data_preprocessing import DataPreprocessor


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
