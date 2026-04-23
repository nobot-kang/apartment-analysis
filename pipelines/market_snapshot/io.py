"""processed parquet 파일 로딩 유틸."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm


def _load_all_trade(processed_dir: Path) -> pd.DataFrame:
    """모든 apt_trade_*.parquet을 로드해 통합한다."""
    files = sorted(processed_dir.glob("apt_trade_[0-9][0-9][0-9][0-9].parquet"))
    if not files:
        logger.warning("apt_trade parquet 파일이 없습니다.")
        return pd.DataFrame()

    dfs = []
    for f in tqdm(files, desc="Loading trade parquets"):
        df = pd.read_parquet(f)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _load_all_rent(processed_dir: Path) -> pd.DataFrame:
    """모든 apt_rent_*.parquet을 로드해 통합한다."""
    files = sorted(processed_dir.glob("apt_rent_[0-9][0-9][0-9][0-9].parquet"))
    if not files:
        logger.warning("apt_rent parquet 파일이 없습니다.")
        return pd.DataFrame()

    dfs = []
    for f in tqdm(files, desc="Loading rent parquets"):
        df = pd.read_parquet(f)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
