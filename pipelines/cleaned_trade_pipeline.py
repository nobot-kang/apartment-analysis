"""정제 매매 데이터셋 생성 파이프라인 (`cleaned_apt_trade`)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import DATA_DIR, PROCESSED_DIR
from pipelines.market_snapshot.outliers.pipeline import build_snapshot_outliers
from pipelines.market_snapshot.preprocess import (
    _add_area_bucket,
    _add_month_column,
    _add_region_columns,
)

PREPROCESSED_PLUS_DIR: Path = DATA_DIR / "preprocessed_plus"

_PHASE1_DTYPES: dict[str, str] = {
    "price": "int32",
    "price_std84": "Int32",
    "price_per_m2": "float32",
    "price_per_py": "float32",
    "area": "float32",
    "floor": "Int16",
    "construction_year": "Int16",
    "age": "Int16",
    "area_repr": "Int16",
}

_REQUIRED_OUTLIER_KEYS = ["date", "price_per_m2", "area_repr", "aptSeq"]

_SGGCD_RE = re.compile(r"^\d{5}$")


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanedTradeManifest:
    build_ts: str
    git_rev: str
    row_counts: dict
    files: list


# ---------------------------------------------------------------------------
# Public loader helpers
# ---------------------------------------------------------------------------


def get_cleaned_trade_manifest(
    data_root: Path | None = None,
    verify_files: bool = True,
    verify_sha256: bool = False,
) -> CleanedTradeManifest:
    """_SUCCESS manifest 존재·유효성만 확인한다. 청크는 로드하지 않는다.

    files[].path 는 data_root 기준 상대경로 (POSIX 구분자).
    실제 검증 경로는 data_root / path 로 해석.

    Raises:
        FileNotFoundError: _SUCCESS 부재.
        RuntimeError: manifest 파싱 실패 또는 파일 size/mtime/sha256 불일치.
    """
    data_root = data_root or DATA_DIR
    success_path = data_root / "processed" / "_SUCCESS"
    if not success_path.exists():
        raise FileNotFoundError(f"_SUCCESS 없음: {success_path}")

    try:
        manifest_data = json.loads(success_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"manifest 파싱 실패: {exc}") from exc

    if verify_files:
        for entry in manifest_data.get("files", []):
            file_path = data_root / entry["path"]
            if not file_path.exists():
                raise RuntimeError(f"manifest 파일 없음: {file_path}")
            stat = file_path.stat()
            if stat.st_size != entry["size_bytes"]:
                raise RuntimeError(
                    f"size_bytes 불일치: {file_path} "
                    f"(manifest={entry['size_bytes']}, actual={stat.st_size})"
                )
            if stat.st_mtime_ns != entry["mtime_ns"]:
                raise RuntimeError(f"mtime_ns 불일치: {file_path}")
            if verify_sha256:
                actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
                if actual != entry["sha256"]:
                    raise RuntimeError(f"sha256 불일치: {file_path}")

    return CleanedTradeManifest(
        build_ts=manifest_data["build_ts"],
        git_rev=manifest_data["git_rev"],
        row_counts=manifest_data["row_counts"],
        files=manifest_data["files"],
    )


def load_cleaned_trade(
    data_root: Path | None = None,
    columns: list[str] | None = None,
    verify_sha256: bool = False,
) -> pd.DataFrame:
    """manifest 검증 후 연도 청크를 concat 해서 반환한다."""
    data_root = data_root or DATA_DIR
    get_cleaned_trade_manifest(
        data_root=data_root, verify_files=True, verify_sha256=verify_sha256
    )
    processed_dir = data_root / "processed"
    files = sorted(processed_dir.glob("cleaned_apt_trade_[0-9][0-9][0-9][0-9].parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Internal dtype / IO helpers
# ---------------------------------------------------------------------------


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in _PHASE1_DTYPES.items():
        if col not in out.columns:
            continue
        if dtype.startswith("Int"):
            out[col] = out[col].round().astype(dtype)
        else:
            out[col] = out[col].astype(dtype)
    if "date" in out.columns:
        out["date"] = out["date"].astype("datetime64[ms]")
    return out


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_project_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _file_info(path: Path, data_root: Path) -> dict:
    stat = path.stat()
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": path.relative_to(data_root).as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256,
    }


def _region_scope(sggcd: str) -> str:
    if _SGGCD_RE.match(sggcd):
        return sggcd
    return {"ALL": "전체", "SEOUL": "서울", "GYEONGGI": "경기"}.get(sggcd, sggcd)


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def _build_cleaned_trade_yearly_summary(
    trade_df: pd.DataFrame,
    explicit_excluded_mask: pd.Series,
    outlier_ids: set,
    filter_summary: pd.DataFrame,
) -> pd.DataFrame:
    """4-stage long-format 연도별 요약 테이블을 생성한다."""
    df = trade_df.copy()
    df["year"] = df["date"].dt.year  # nullable — NaT rows drop in groupby
    df["_exex"] = explicit_excluded_mask.values
    df["_outlier"] = df["row_id"].isin(outlier_ids)

    def _count_stages(sub: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        acd = sub.groupby(keys).size().rename("after_cancel_direct")
        aex = sub[~sub["_exex"]].groupby(keys).size().rename("after_explicit_excluded")
        aout = (
            sub[~sub["_exex"] & ~sub["_outlier"]].groupby(keys).size().rename("after_outlier")
        )
        return pd.concat([acd, aex, aout], axis=1).reset_index().fillna(0)

    per_sggcd = _count_stages(df, ["year", "sggCd"])

    agg_rows: list[pd.DataFrame] = []
    for code, rt in [("ALL", None), ("SEOUL", "서울"), ("GYEONGGI", "경기")]:
        sub = df if rt is None else df[df["region_type"].eq(rt)]
        agg = _count_stages(sub, ["year"])
        agg["sggCd"] = code
        agg_rows.append(agg)

    all_counts = pd.concat([per_sggcd, *agg_rows], ignore_index=True)

    raw_df = (
        filter_summary[["year", "sggCd", "total_trade_count"]]
        .rename(columns={"total_trade_count": "raw"})
    )
    all_counts = all_counts.merge(raw_df, on=["year", "sggCd"], how="left")

    stages = ["raw", "after_cancel_direct", "after_explicit_excluded", "after_outlier"]
    rows: list[dict] = []
    for _, r in all_counts.iterrows():
        if pd.isna(r.get("year")):
            continue
        raw = int(r["raw"]) if pd.notna(r.get("raw")) else 0
        for i, stage in enumerate(stages):
            count = int(r[stage]) if pd.notna(r.get(stage)) else 0
            pct = count / raw * 100 if raw > 0 else float("nan")
            removed = 0 if i == 0 else (int(r[stages[i - 1]]) if pd.notna(r.get(stages[i - 1])) else 0) - count
            sggcd = str(r["sggCd"])
            rows.append({
                "year": int(r["year"]),
                "region_scope": _region_scope(sggcd),
                "sggCd": sggcd,
                "stage": stage,
                "count": count,
                "pct_of_raw": pct,
                "removed_count": removed,
            })
    return pd.DataFrame(rows)


def _build_outlier_reason_summary(
    trade_df: pd.DataFrame,
    verdicts: pd.DataFrame,
) -> pd.DataFrame:
    """outlier_reason 별 연도·지역 long-format 요약을 생성한다."""
    meta = trade_df[["row_id", "date", "sggCd", "region_type"]].copy()
    meta["year"] = meta["date"].dt.year  # nullable — NaT rows drop in groupby

    outlier_v = verdicts[verdicts["is_outlier"]].copy()
    merged = outlier_v.merge(meta, on="row_id", how="left")

    if merged.empty:
        return pd.DataFrame(
            columns=["year", "region_scope", "sggCd", "outlier_reason", "count"]
        )

    per_sggcd = (
        merged.groupby(["year", "sggCd", "outlier_reason"])
        .size()
        .reset_index(name="count")
    )

    agg_rows: list[pd.DataFrame] = []
    for code, rt in [("ALL", None), ("SEOUL", "서울"), ("GYEONGGI", "경기")]:
        sub = merged if rt is None else merged[merged["region_type"].eq(rt)]
        agg = sub.groupby(["year", "outlier_reason"]).size().reset_index(name="count")
        agg["sggCd"] = code
        agg_rows.append(agg)

    all_rows = pd.concat([per_sggcd, *agg_rows], ignore_index=True)
    all_rows["region_scope"] = all_rows["sggCd"].apply(_region_scope)
    return all_rows[["year", "region_scope", "sggCd", "outlier_reason", "count"]]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_input_count(
    trade_df: pd.DataFrame,
    filter_summary: pd.DataFrame,
    years: list[int] | None,
) -> None:
    if "after_cancel_direct_count" not in filter_summary.columns:
        logger.warning(
            "trade_filter_yearly_summary 에 after_cancel_direct_count 컬럼이 없습니다. "
            "정합성 게이트를 건너뜁니다. preprocess_trade() 를 재실행하세요."
        )
        return

    trade_years = trade_df["date"].dt.year.dropna().astype(int).unique()
    sggcd_rows = filter_summary[filter_summary["sggCd"].str.match(r"^\d{5}$")]
    if years:
        sggcd_rows = sggcd_rows[sggcd_rows["year"].isin(years)]
    else:
        sggcd_rows = sggcd_rows[sggcd_rows["year"].isin(trade_years)]

    expected = int(sggcd_rows["after_cancel_direct_count"].sum())
    actual = len(trade_df)
    if expected != actual:
        raise RuntimeError(
            f"입력 행수 불일치: trade_filter_yearly_summary 예상={expected:,}, "
            f"실제={actual:,}. trade_filter_yearly_summary.parquet 가 stale 할 수 있습니다."
        )
    logger.info(f"정합성 게이트 통과: after_cancel_direct_count={expected:,}")


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def build_cleaned_trade(
    processed_dir: Path | None = None,
    preprocessed_plus_dir: Path | None = None,
    data_root: Path | None = None,
    include_combined: bool = True,
    years: list[int] | None = None,
    variant: str | None = None,
) -> None:
    """cleaned_apt_trade 데이터셋을 생성하고 _SUCCESS manifest 를 기록한다.

    Steps:
      1. Preflight: input 로드 + 정합성 검증
      2. 분류 마스크 계산 + partition completeness assertion
      3. A-3 이상치 탐지 (return_verdicts=True)
      4. cleaned DataFrame 구성
      5. dtype 최적화
      6. 요약 산출물 계산
      7. 모든 *.tmp 작성
      8. _SUCCESS unlink → replace → _SUCCESS 기록
    """
    data_root = data_root or DATA_DIR
    processed_dir = processed_dir or PROCESSED_DIR
    preprocessed_plus_dir = preprocessed_plus_dir or PREPROCESSED_PLUS_DIR
    preprocessed_plus_dir.mkdir(parents=True, exist_ok=True)

    success_path = processed_dir / "_SUCCESS"

    # ------------------------------------------------------------------
    # Step 1: Preflight
    # ------------------------------------------------------------------
    logger.info("=== cleaned_apt_trade 빌드 시작 ===")

    filter_summary_path = preprocessed_plus_dir / "trade_filter_yearly_summary.parquet"
    if not filter_summary_path.exists():
        raise RuntimeError(
            f"trade_filter_yearly_summary.parquet 없음: {filter_summary_path}. "
            "preprocess_trade() 를 먼저 실행하세요."
        )
    filter_summary = pd.read_parquet(filter_summary_path)

    input_files = sorted(processed_dir.glob("apt_trade_[0-9][0-9][0-9][0-9].parquet"))
    if years:
        input_files = [
            f for f in input_files
            if int(f.stem.rsplit("_", 1)[-1]) in years
        ]
    if not input_files:
        raise RuntimeError(
            f"입력 파일 없음: {processed_dir}/apt_trade_[0-9][0-9][0-9][0-9].parquet"
        )

    frames = [pd.read_parquet(f) for f in input_files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("로드된 apt_trade 데이터가 없습니다.")

    trade_df = pd.concat(frames, ignore_index=True)
    logger.info(f"입력 로드 완료: {len(trade_df):,}건 ({len(input_files)}개 파일)")

    trade_df["row_id"] = np.arange(len(trade_df), dtype=np.int64)
    trade_df = _add_region_columns(trade_df)
    trade_df = _add_area_bucket(trade_df)
    trade_df = _add_month_column(trade_df)

    _validate_input_count(trade_df, filter_summary, years)

    # ------------------------------------------------------------------
    # Step 2: Classification masks (A-3 precedence)
    # ------------------------------------------------------------------
    required_key_mask = trade_df[_REQUIRED_OUTLIER_KEYS].notna().all(axis=1)
    floor_eq_1 = pd.to_numeric(trade_df["floor"], errors="coerce").eq(1)

    explicit_excluded_mask: pd.Series = ~required_key_mask
    floor1_mask: pd.Series = required_key_mask & floor_eq_1
    evaluated_mask: pd.Series = required_key_mask & ~floor_eq_1

    assert (explicit_excluded_mask & floor1_mask).sum() == 0
    assert (explicit_excluded_mask & evaluated_mask).sum() == 0
    assert (floor1_mask & evaluated_mask).sum() == 0
    assert (explicit_excluded_mask | floor1_mask | evaluated_mask).all()

    # ------------------------------------------------------------------
    # Step 3: A-3 outlier evaluation
    # ------------------------------------------------------------------
    logger.info("A-3 이상치 탐지 실행 중...")
    _outliers_df, _market_price_df, verdicts = build_snapshot_outliers(
        trade_df, return_verdicts=True
    )

    outlier_ids: set = set(verdicts.loc[verdicts["is_outlier"], "row_id"].tolist())
    explicit_excluded_ids: set = set(
        trade_df.loc[explicit_excluded_mask, "row_id"].tolist()
    )
    all_ids: set = set(trade_df["row_id"].tolist())
    cleaned_ids: set = all_ids - outlier_ids - explicit_excluded_ids

    if all_ids != (cleaned_ids | outlier_ids | explicit_excluded_ids):
        raise RuntimeError("partition completeness 실패: id 집합 불일치")
    if cleaned_ids & outlier_ids:
        raise RuntimeError("partition completeness 실패: cleaned ∩ outlier ≠ ∅")
    if cleaned_ids & explicit_excluded_ids:
        raise RuntimeError("partition completeness 실패: cleaned ∩ explicit_excluded ≠ ∅")

    logger.info(
        f"분류 완료: input={len(all_ids):,} / cleaned={len(cleaned_ids):,} / "
        f"outlier={len(outlier_ids):,} / explicit_excluded={len(explicit_excluded_ids):,}"
    )

    # ------------------------------------------------------------------
    # Step 4: Build cleaned DataFrame
    # ------------------------------------------------------------------
    cleaned = trade_df[~trade_df["row_id"].isin(outlier_ids | explicit_excluded_ids)].copy()

    # Remove internal/diagnostic columns from canonical artifact
    _drop = [
        "row_id", "sggCd", "region_name", "region_type", "area_bucket", "month", "_exex", "_outlier",
    ]
    if variant != "with-diagnostics":
        diag = [c for c in cleaned.columns if c.startswith("spread_") or c.startswith("ref_price_") or c in ("band_pct", "band_abs_m2")]
        _drop.extend(diag)
    cleaned = cleaned.drop(columns=[c for c in _drop if c in cleaned.columns])

    if variant == "exclude-floor1":
        floor_num = pd.to_numeric(cleaned.get("floor", pd.Series(dtype=float)), errors="coerce")
        cleaned = cleaned[floor_num != 1].copy()

    cleaned = _optimize_dtypes(cleaned)

    # ------------------------------------------------------------------
    # Step 5: Summary artifacts
    # ------------------------------------------------------------------
    logger.info("요약 산출물 생성 중...")
    trade_df["year"] = trade_df["date"].dt.year  # nullable — NaT rows excluded in groupby
    yearly_summary = _build_cleaned_trade_yearly_summary(
        trade_df, explicit_excluded_mask, outlier_ids, filter_summary
    )
    reason_summary = _build_outlier_reason_summary(trade_df, verdicts)

    # ------------------------------------------------------------------
    # Step 6: Handle variant (writes to variants/ and returns early)
    # ------------------------------------------------------------------
    if variant:
        variant_dir = processed_dir / "variants"
        variant_dir.mkdir(parents=True, exist_ok=True)
        prefix_map = {
            "with-diagnostics": "cleaned_apt_trade_with_diag",
            "exclude-floor1": "cleaned_apt_trade_no_floor1",
        }
        out_prefix = prefix_map.get(variant, f"cleaned_apt_trade_{variant}")
        date_col = pd.to_datetime(cleaned["date"])
        cleaned["_yr"] = date_col.dt.year
        for yr, chunk in cleaned.groupby("_yr"):
            chunk = chunk.drop(columns=["_yr"])
            (variant_dir / f"{out_prefix}_{yr}.parquet").write_bytes(b"")  # placeholder
            chunk.to_parquet(variant_dir / f"{out_prefix}_{yr}.parquet", index=False)
        logger.info(f"variant '{variant}' 산출물 완료: {variant_dir}")
        return

    # ------------------------------------------------------------------
    # Step 7: Write *.tmp for all canonical artifacts
    # ------------------------------------------------------------------
    logger.info("임시 파일 작성 중...")

    date_col = pd.to_datetime(cleaned["date"])
    cleaned["_yr"] = date_col.dt.year

    chunk_targets: list[tuple[Path, pd.DataFrame]] = []
    for yr, chunk in cleaned.groupby("_yr"):
        target = processed_dir / f"cleaned_apt_trade_{yr}.parquet"
        tmp = target.with_suffix(".parquet.tmp")
        chunk.drop(columns=["_yr"]).to_parquet(tmp, index=False)
        chunk_targets.append((target, tmp))

    combined_target: Path | None = None
    combined_tmp: Path | None = None
    if include_combined:
        combined_target = processed_dir / "cleaned_apt_trade.parquet"
        combined_tmp = combined_target.with_suffix(".parquet.tmp")
        cleaned.drop(columns=["_yr"]).to_parquet(combined_tmp, index=False)

    summary_target = preprocessed_plus_dir / "cleaned_trade_yearly_summary.parquet"
    summary_tmp = summary_target.with_suffix(".parquet.tmp")
    yearly_summary.to_parquet(summary_tmp, index=False)

    reason_target = preprocessed_plus_dir / "cleaned_trade_outlier_reason_summary.parquet"
    reason_tmp = reason_target.with_suffix(".parquet.tmp")
    reason_summary.to_parquet(reason_tmp, index=False)

    # ------------------------------------------------------------------
    # Step 8: Unlink _SUCCESS → atomically replace → write _SUCCESS
    # ------------------------------------------------------------------
    if success_path.exists():
        success_path.unlink()

    for target, tmp in chunk_targets:
        tmp.replace(target)
    if combined_target is not None and combined_tmp is not None:
        combined_tmp.replace(combined_target)
    summary_tmp.replace(summary_target)
    reason_tmp.replace(reason_target)

    manifest_files: list[dict] = []
    for target, _ in chunk_targets:
        manifest_files.append(_file_info(target, data_root))
    if combined_target is not None:
        manifest_files.append(_file_info(combined_target, data_root))
    manifest_files.append(_file_info(summary_target, data_root))
    manifest_files.append(_file_info(reason_target, data_root))

    manifest = {
        "build_ts": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "row_counts": {
            "input": len(all_ids),
            "cleaned": len(cleaned_ids),
            "outlier": len(outlier_ids),
            "explicit_excluded": len(explicit_excluded_ids),
        },
        "files": manifest_files,
    }
    success_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("=== cleaned_apt_trade 빌드 완료. _SUCCESS 기록 완료 ===")
