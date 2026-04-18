"""공용 smoothing helper: 로그 공간 중앙값 이동평균."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.market_snapshot.config import PATH_WINDOW_MONTHS, PATH_MIN_PERIODS


def _centered_rolling_median(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    result = s.rolling(window, center=True, min_periods=min_periods).median()
    result = result.ffill().bfill()
    return result


def smooth_log_series(grp: pd.DataFrame, val_col: str, out_col: str) -> pd.DataFrame:
    """val_col 을 로그 변환 후 centered rolling median 으로 스무딩해 out_col 에 저장한다."""
    grp = grp.sort_values("month").copy()
    log_vals = np.where(grp[val_col] > 0, np.log(grp[val_col]), np.nan)
    smoothed = _centered_rolling_median(
        pd.Series(log_vals, index=grp.index), PATH_WINDOW_MONTHS, PATH_MIN_PERIODS
    )
    grp[out_col] = np.exp(smoothed)
    return grp
