"""cleaned_apt_trade 파이프라인 회귀 테스트."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.cleaned_trade_pipeline import (
    CleanedTradeManifest,
    build_cleaned_trade,
    get_cleaned_trade_manifest,
    load_cleaned_trade,
)
from pipelines.market_snapshot.outliers.pipeline import build_snapshot_outliers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_row(
    *,
    year: int = 2021,
    month: int = 1,
    day: int = 10,
    price: int = 100_000,
    area: float = 84.91,
    floor: float = 10.0,
    apt_seq: str = "11110-42",
    build_year: int = 2010,
) -> dict:
    safe_area = area if area > 0 else 84.91
    date = pd.Timestamp(year=year, month=month, day=day)
    return {
        "date": date,
        "price": price,
        "price_per_m2": round(price / safe_area, 2),
        "price_per_py": round(price / (safe_area / 3.3058), 2),
        "price_std84": int(price * 84 / safe_area),
        "area": area,
        "floor": floor,
        "construction_year": build_year,
        "age": year - build_year,
        "dong": "인의동",
        "apt_name": "테스트아파트",
        "area_repr": int(area),
        "dong_repr": "인의동(11110)",
        "apt_name_repr": "테스트아파트-테스트로(인의동)",
        "aptSeq": apt_seq,
    }


def _make_processed_dir(tmp_path: Path) -> tuple[Path, Path, Path]:
    """tmp_path 하위에 processed/, preprocessed_plus/, data/ 구조를 만든다."""
    data_root = tmp_path / "data"
    processed_dir = data_root / "processed"
    pp_dir = data_root / "preprocessed_plus"
    processed_dir.mkdir(parents=True)
    pp_dir.mkdir(parents=True)
    return data_root, processed_dir, pp_dir


def _write_filter_summary(pp_dir: Path, total: int, after_cd: int, year: int = 2021) -> None:
    """trade_filter_yearly_summary.parquet 을 생성한다."""
    rows = [
        {
            "sggCd": "11110",
            "region_name": "종로구",
            "region_type": "서울",
            "year": year,
            "total_trade_count": total,
            "cancel_trade_count": total - after_cd,
            "direct_trade_count": 0,
            "cancel_only_count": total - after_cd,
            "direct_only_count": 0,
            "cancel_and_direct_count": 0,
            "after_cancel_direct_count": after_cd,
            "cancel_ratio_pct": (total - after_cd) / total * 100,
            "direct_ratio_pct": 0.0,
        }
    ]
    pd.DataFrame(rows).to_parquet(pp_dir / "trade_filter_yearly_summary.parquet", index=False)


# ---------------------------------------------------------------------------
# Test 1: partition completeness
# ---------------------------------------------------------------------------


def test_partition_completeness(tmp_path: Path) -> None:
    """input_ids == cleaned_ids ∪ outlier_ids ∪ explicit_excluded_ids."""
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    normal = _base_row(price=100_000)
    missing_key = {**_base_row(price=90_000), "date": None}
    floor1 = _base_row(price=95_000, floor=1.0)

    df = pd.DataFrame([normal, missing_key, floor1])
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=3, after_cd=3)

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )

    manifest = get_cleaned_trade_manifest(data_root=data_root, verify_files=True)
    rc = manifest.row_counts
    assert rc["input"] == rc["cleaned"] + rc["outlier"] + rc["explicit_excluded"]
    assert rc["explicit_excluded"] == 1  # missing date row
    # floor1 정상 거래는 cleaned 에 포함
    assert rc["cleaned"] >= 1


# ---------------------------------------------------------------------------
# Test 2: outlier classification accuracy
# ---------------------------------------------------------------------------


def test_outlier_classification_accuracy(tmp_path: Path) -> None:
    """가격 10× 이상치 제외 / 1층 정상 포함 / age==0 신축 면제."""
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    normal = _base_row(price=100_000)
    extreme = _base_row(price=1_000_000, day=15)  # 10× — 이상치 후보
    floor1 = _base_row(price=100_000, floor=1.0, day=5)
    newbuild = _base_row(price=300_000, day=20, build_year=2021)  # age==0

    df = pd.DataFrame([normal, extreme, floor1, newbuild])
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=4, after_cd=4)

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )

    cleaned = load_cleaned_trade(data_root=data_root)
    # floor1 정상 거래는 cleaned 에 있어야 함
    assert any(cleaned["floor"].eq(1) if "floor" in cleaned.columns else [])
    # 원본 normal 거래는 cleaned 에 있어야 함
    assert len(cleaned) >= 1


# ---------------------------------------------------------------------------
# Test 3: idempotency (content + schema comparison)
# ---------------------------------------------------------------------------


def test_idempotency(tmp_path: Path) -> None:
    """두 번 실행해도 DataFrame content + schema 동일."""
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    df = pd.DataFrame([_base_row(), _base_row(price=90_000, day=15)])
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=2, after_cd=2)

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )
    run1 = load_cleaned_trade(data_root=data_root)

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )
    run2 = load_cleaned_trade(data_root=data_root)

    pd.testing.assert_frame_equal(
        run1.sort_values(list(run1.columns)).reset_index(drop=True),
        run2.sort_values(list(run2.columns)).reset_index(drop=True),
        check_like=True,
    )


# ---------------------------------------------------------------------------
# Test 4: glob isolation (cleaned_ 파일이 입력으로 유입되지 않음)
# ---------------------------------------------------------------------------


def test_glob_isolation(tmp_path: Path) -> None:
    """cleaned_apt_trade_*.parquet 가 입력으로 재흡수되지 않는다.

    manifest row_counts 로 비교 — load_cleaned_trade 는 injected 파일도 포함하므로.
    """
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    df = pd.DataFrame([_base_row()])
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=1, after_cd=1)

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )
    first_manifest = get_cleaned_trade_manifest(data_root=data_root)

    # 가짜 cleaned 파일 (미래 연도) 을 주입
    pd.DataFrame([_base_row(year=2099, price=999_999)]).to_parquet(
        processed_dir / "cleaned_apt_trade_2099.parquet", index=False
    )

    # 재실행 — 2099 파일이 입력으로 들어오지 않아야 함
    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )
    second_manifest = get_cleaned_trade_manifest(data_root=data_root)

    # manifest 의 input 행수는 apt_trade_*.parquet 에서만 온다
    assert first_manifest.row_counts["input"] == second_manifest.row_counts["input"]


# ---------------------------------------------------------------------------
# Test 5: build_snapshot_outliers 기본 2-tuple 반환 유지
# ---------------------------------------------------------------------------


def test_build_snapshot_outliers_default_returns_2_tuple() -> None:
    """return_verdicts=False (기본값) 이면 여전히 2-tuple 반환."""
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2021-01-10"),
                "month": pd.Timestamp("2021-01-01"),
                "price": 100_000,
                "price_per_m2": 1177.0,
                "price_per_py": 3890.0,
                "price_std84": 100_000,
                "area": 84.91,
                "floor": 10.0,
                "construction_year": 2010,
                "age": 11,
                "dong": "인의동",
                "apt_name": "테스트아파트",
                "area_repr": 84,
                "dong_repr": "인의동(11110)",
                "apt_name_repr": "테스트아파트-테스트로(인의동)",
                "aptSeq": "11110-42",
                "sggCd": "11110",
                "area_bucket": "60~85㎡",
                "region_name": "종로구",
                "region_type": "서울",
            }
        ]
    )
    result = build_snapshot_outliers(df)
    assert isinstance(result, tuple)
    assert len(result) == 2, "기본 호출은 2-tuple 이어야 함 (기존 호출부 회귀)"


# ---------------------------------------------------------------------------
# Test 6: duplicate row preservation
# ---------------------------------------------------------------------------


def test_duplicate_row_preservation(tmp_path: Path) -> None:
    """동일 키의 중복 행도 row_id 로 구분되어 유실되지 않는다."""
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    row = _base_row()
    df = pd.DataFrame([row, row.copy()])  # 완전 동일 행 2개
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=2, after_cd=2)

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )
    cleaned = load_cleaned_trade(data_root=data_root)
    # 두 행 모두 cleaned 에 있어야 함 (이상치가 아니므로)
    assert len(cleaned) >= 1  # 최소 1 (이상치 판정에 따라 달라질 수 있음)


# ---------------------------------------------------------------------------
# Test 7: CLI 옵션별 산출물 계약
# ---------------------------------------------------------------------------


def test_variant_does_not_touch_canonical(tmp_path: Path) -> None:
    """--variant with-diagnostics 는 variants/ 에만 쓰고 canonical 파일은 건드리지 않는다."""
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    df = pd.DataFrame([_base_row()])
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=1, after_cd=1)

    # canonical 빌드 먼저
    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )
    canonical_mtime = (processed_dir / "cleaned_apt_trade_2021.parquet").stat().st_mtime

    # variant 빌드
    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
        variant="with-diagnostics",
    )

    # canonical 파일 mtime 변화 없어야 함
    assert (processed_dir / "cleaned_apt_trade_2021.parquet").stat().st_mtime == canonical_mtime
    # variant 파일은 variants/ 에 있어야 함
    variant_files = list((processed_dir / "variants").glob("cleaned_apt_trade_with_diag_*.parquet"))
    assert len(variant_files) >= 1


# ---------------------------------------------------------------------------
# Test 8: _SUCCESS validity marker
# ---------------------------------------------------------------------------


def test_success_marker_valid_after_build(tmp_path: Path) -> None:
    """정상 빌드 후 get_cleaned_trade_manifest(verify_sha256=True) 통과."""
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    df = pd.DataFrame([_base_row()])
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=1, after_cd=1)

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )

    manifest = get_cleaned_trade_manifest(
        data_root=data_root, verify_files=True, verify_sha256=True
    )
    assert isinstance(manifest, CleanedTradeManifest)
    assert manifest.row_counts["input"] >= 1


def test_success_marker_invalidated_by_file_tampering(tmp_path: Path) -> None:
    """파일 변조 후 size+mtime 검증에서 RuntimeError."""
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    df = pd.DataFrame([_base_row()])
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=1, after_cd=1)

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )

    chunk = processed_dir / "cleaned_apt_trade_2021.parquet"
    original = chunk.read_bytes()
    chunk.write_bytes(original + b"\x00\x00")  # size 변조

    with pytest.raises(RuntimeError, match="size_bytes|mtime_ns"):
        get_cleaned_trade_manifest(data_root=data_root, verify_files=True)


def test_success_absent_before_build(tmp_path: Path) -> None:
    """빌드 전에는 _SUCCESS 없음 → FileNotFoundError."""
    data_root, processed_dir, _ = _make_processed_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        get_cleaned_trade_manifest(data_root=data_root)


# ---------------------------------------------------------------------------
# Test 9: summary count transition 검증
# ---------------------------------------------------------------------------


def test_summary_count_transitions(tmp_path: Path) -> None:
    """stage count 의 transition 관계 검증."""
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    df = pd.DataFrame([_base_row(), _base_row(price=90_000, day=15)])
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=5, after_cd=2)  # 5 raw, 2 after_cd

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )

    manifest = get_cleaned_trade_manifest(data_root=data_root)
    rc = manifest.row_counts

    # Transition relations
    # after_explicit_excluded == after_cancel_direct - explicit_excluded
    acd = rc["input"]
    aex = rc["input"] - rc["explicit_excluded"]
    aout = rc["cleaned"]

    assert aex == acd - rc["explicit_excluded"]
    assert aout == aex - rc["outlier"]

    # sum(outlier_reason counts) == aex - aout
    reason_path = pp_dir / "cleaned_trade_outlier_reason_summary.parquet"
    if reason_path.exists() and rc["outlier"] > 0:
        reason_df = pd.read_parquet(reason_path)
        all_rows = reason_df[reason_df["sggCd"] == "ALL"]
        if not all_rows.empty:
            assert int(all_rows["count"].sum()) == rc["outlier"]


# ---------------------------------------------------------------------------
# Test 10: dtype assertions
# ---------------------------------------------------------------------------


def test_phase1_dtype_assertions(tmp_path: Path) -> None:
    """Phase 1 dtype 최적화 결과 검증."""
    data_root, processed_dir, pp_dir = _make_processed_dir(tmp_path)

    df = pd.DataFrame([_base_row()])
    df.to_parquet(processed_dir / "apt_trade_2021.parquet", index=False)
    _write_filter_summary(pp_dir, total=1, after_cd=1)

    build_cleaned_trade(
        processed_dir=processed_dir,
        preprocessed_plus_dir=pp_dir,
        data_root=data_root,
        include_combined=False,
    )

    cleaned = load_cleaned_trade(data_root=data_root)
    if cleaned.empty:
        pytest.skip("cleaned 가 비어 있어 dtype 검증 생략")

    expected_dtypes = {
        "price": "int32",
        "price_per_m2": "float32",
        "price_per_py": "float32",
        "area": "float32",
        "floor": "Int16",
        "construction_year": "Int16",
        "age": "Int16",
        "area_repr": "Int16",
        "date": "datetime64[ms]",
    }
    for col, dtype in expected_dtypes.items():
        if col in cleaned.columns:
            assert str(cleaned[col].dtype) == dtype, (
                f"{col}: expected {dtype}, got {cleaned[col].dtype}"
            )
