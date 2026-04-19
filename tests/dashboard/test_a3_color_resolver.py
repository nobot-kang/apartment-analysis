"""Unit tests for _resolve_color_col helper (A-3 outlier section)."""

import pytest

# Import the pure helper without triggering Streamlit's module-level side-effects.
# The function lives at module top level so we can import it directly.
from dashboard.pages.page_00_market_snapshot_diagnostics import _resolve_color_col

COLS_WITH_DIRECTION = {"outlier_direction", "판정사유", "month"}
COLS_WITHOUT_DIRECTION = {"판정사유", "month"}


@pytest.mark.parametrize(
    "mode, reason_key, columns, expected",
    [
        # auto — reason_key is None  → 판정사유 (no fallback)
        ("auto", None, COLS_WITH_DIRECTION, ("판정사유", False)),
        # auto — reason_key set, column present → outlier_direction (no fallback)
        ("auto", "sanity_error", COLS_WITH_DIRECTION, ("outlier_direction", False)),
        # auto — reason_key set, column ABSENT → fallback to 판정사유 (core regression guard)
        ("auto", "sanity_error", COLS_WITHOUT_DIRECTION, ("판정사유", True)),
        # direction — column absent → fallback
        ("direction", None, COLS_WITHOUT_DIRECTION, ("판정사유", True)),
        # reason — always 판정사유 regardless of columns (new UI branch, normal path)
        ("reason", None, COLS_WITHOUT_DIRECTION, ("판정사유", False)),
        ("reason", "sanity_error", COLS_WITH_DIRECTION, ("판정사유", False)),
        # direction — column present (new UI branch, normal path)
        ("direction", None, COLS_WITH_DIRECTION, ("outlier_direction", False)),
        ("direction", "sanity_error", COLS_WITH_DIRECTION, ("outlier_direction", False)),
    ],
)
def test_resolve_color_col(mode, reason_key, columns, expected):
    assert _resolve_color_col(mode, reason_key, columns) == expected
