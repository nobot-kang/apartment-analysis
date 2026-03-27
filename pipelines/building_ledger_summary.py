"""Summarize title and recap title building-ledger datasets into one lookup table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import MOLIT_RAW_DIR, PROCESSED_DIR

TITLE_PATH_NAME = "building_ledger_all.parquet"
RECAP_PATH_NAME = "building_ledger_recap_all.parquet"

PARKING_COMPONENT_ALIASES = {
    "indoor_mech": ["indrMechUtCnt", "indrMechUtcnt"],
    "outdoor_mech": ["oudrMechUtCnt", "oudrMechUtcnt"],
    "indoor_auto": ["indrAutoUtCnt", "indrAutoUtcnt"],
    "outdoor_auto": ["oudrAutoUtCnt", "oudrAutoUtcnt"],
}

RAW_TO_STANDARD = {
    "apt_name_ledger": ["bldNm"],
    "completion_date": ["useAprDay"],
    "created_day": ["crtnDay"],
    "land_area": ["platArea"],
    "floor_area_ratio_total_area": ["vlRatEstmTotArea"],
    "total_area": ["totArea"],
    "floor_area_ratio": ["vlRat"],
    "building_coverage_ratio": ["bcRat"],
    "household_count": ["hhldCnt"],
    "total_parking_count_api": ["totPkngCnt"],
    "ground_floor_count": ["grndFlrCnt"],
    "underground_floor_count": ["ugrndFlrCnt"],
    "address": ["platPlc"],
    "sigungu_code": ["sigunguCd"],
    "bjdong_code": ["bjdongCd"],
    "regstr_kind_name": ["regstrKindCdNm"],
    "regstr_type_code": ["regstrGbCd"],
}

NUMERIC_FIELDS = {
    "land_area",
    "floor_area_ratio_total_area",
    "total_area",
    "floor_area_ratio",
    "building_coverage_ratio",
    "household_count",
    "total_parking_count_api",
    "ground_floor_count",
    "underground_floor_count",
}


class BuildingLedgerSummarizer:
    """Build a single apartment lookup table from title and recap-title ledgers."""

    def __init__(
        self,
        raw_dir: str | Path | None = None,
        processed_dir: str | Path | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir) if raw_dir else MOLIT_RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.title_raw_path = self.raw_dir / "building_ledger" / TITLE_PATH_NAME
        self.recap_raw_path = self.raw_dir / "building_ledger" / RECAP_PATH_NAME
        self.out_path = self.processed_dir / "apartment_info.parquet"

    @staticmethod
    def _clean_date(series: pd.Series) -> pd.Series:
        text = series.astype("string").str.replace(".0", "", regex=False).str.strip()
        text = text.replace(["0", "", "None", "nan", "NaN", "<NA>"], pd.NA)
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")

    @staticmethod
    def _pick_column(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
        for column in aliases:
            if column in df.columns:
                return df[column]
        return pd.Series(pd.NA, index=df.index, dtype="object")

    @staticmethod
    def _valid_text(series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip().ne("") & series.notna()

    @staticmethod
    def _valid_numeric(series: pd.Series, *, lower: float | None = None, upper: float | None = None) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        mask = values.notna()
        if lower is not None:
            mask &= values > lower
        if upper is not None:
            mask &= values <= upper
        return mask

    @staticmethod
    def _choose_preferred(
        preferred: pd.Series,
        fallback: pd.Series,
        *,
        preferred_label: str,
        fallback_label: str,
        valid_mask_preferred: pd.Series,
        valid_mask_fallback: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        selected = fallback.astype("object").copy()
        selected[valid_mask_preferred] = preferred[valid_mask_preferred]
        selected[~valid_mask_preferred & ~valid_mask_fallback] = pd.NA

        source = pd.Series(pd.NA, index=preferred.index, dtype="string")
        source[valid_mask_preferred] = preferred_label
        source[~valid_mask_preferred & valid_mask_fallback] = fallback_label
        return selected, source

    def _load_raw(self, path: Path, label: str) -> pd.DataFrame:
        if not path.exists():
            logger.warning("{} raw parquet is missing: {}", label, path)
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if df.empty:
            logger.warning("{} raw parquet is empty: {}", label, path)
            return df
        if "aptSeq" not in df.columns:
            logger.warning("{} raw parquet has no aptSeq column: {}", label, path)
            return pd.DataFrame()
        return df.dropna(subset=["aptSeq"]).copy()

    def _derive_parking_components(self, df: pd.DataFrame) -> pd.Series:
        component_frames: list[pd.Series] = []
        for aliases in PARKING_COMPONENT_ALIASES.values():
            series = self._pick_column(df, aliases)
            component_frames.append(pd.to_numeric(series, errors="coerce"))
        if not component_frames:
            return pd.Series(np.nan, index=df.index)
        component_df = pd.concat(component_frames, axis=1)
        any_present = component_df.notna().any(axis=1)
        total = component_df.fillna(0).sum(axis=1)
        return total.where(any_present, np.nan)

    def _summarize_source(self, df: pd.DataFrame, source_name: str, expected_kind: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["aptSeq"])

        working = df.copy()
        working["aptSeq"] = working["aptSeq"].astype("string")
        working["_priority_regstr"] = np.where(working.get("regstrGbCd", pd.Series(index=working.index)).astype("string") == "2", 0, 1)
        regstr_kind = working.get("regstrKindCdNm", pd.Series("", index=working.index)).astype("string")
        working["_priority_kind"] = np.where(regstr_kind.str.contains(expected_kind, na=False), 0, 1)
        created_day = self._clean_date(self._pick_column(working, ["crtnDay"]))
        working["_created_day"] = created_day
        working = working.sort_values(["aptSeq", "_priority_kind", "_priority_regstr", "_created_day"], ascending=[True, True, True, False])
        working = working.drop_duplicates(subset=["aptSeq"], keep="first").copy()

        result = pd.DataFrame({"aptSeq": working["aptSeq"]})
        for target, aliases in RAW_TO_STANDARD.items():
            series = self._pick_column(working, aliases)
            if target in NUMERIC_FIELDS:
                result[f"{source_name}_{target}"] = pd.to_numeric(series, errors="coerce")
            elif target in {"completion_date", "created_day"}:
                result[f"{source_name}_{target}"] = self._clean_date(series)
            else:
                result[f"{source_name}_{target}"] = series.astype("string").str.strip()

        parking_components = self._derive_parking_components(working)
        parking_api = result[f"{source_name}_total_parking_count_api"]
        parking_api_valid = self._valid_numeric(parking_api, lower=0, upper=30000)
        parking_component_valid = self._valid_numeric(parking_components, lower=0, upper=30000)
        parking_candidate = parking_components.copy()
        parking_candidate[parking_api_valid] = parking_api[parking_api_valid]
        parking_candidate[~parking_api_valid & ~parking_component_valid] = np.nan
        parking_source = pd.Series(pd.NA, index=result.index, dtype="string")
        parking_source[parking_api_valid] = "api"
        parking_source[~parking_api_valid & parking_component_valid] = "components"

        result[f"{source_name}_parking_component_count"] = parking_components
        result[f"{source_name}_total_parking_count"] = parking_candidate
        result[f"{source_name}_parking_count_source"] = parking_source
        result[f"{source_name}_parking_per_household"] = parking_candidate / result[f"{source_name}_household_count"].replace(0, np.nan)
        return result

    def _merge_sources(self, title_df: pd.DataFrame, recap_df: pd.DataFrame) -> pd.DataFrame:
        if title_df.empty and recap_df.empty:
            return pd.DataFrame()
        if title_df.empty:
            return recap_df.copy()
        if recap_df.empty:
            return title_df.copy()
        return title_df.merge(recap_df, on="aptSeq", how="outer")

    def summarize(self) -> pd.DataFrame:
        title_raw = self._load_raw(self.title_raw_path, "Title")
        recap_raw = self._load_raw(self.recap_raw_path, "Recap title")
        if title_raw.empty and recap_raw.empty:
            logger.error("No building-ledger raw files were found.")
            return pd.DataFrame()

        title_summary = self._summarize_source(title_raw, "title", "표제부")
        recap_summary = self._summarize_source(recap_raw, "recap", "총괄")
        merged = self._merge_sources(title_summary, recap_summary)
        if merged.empty:
            logger.error("No summary rows were produced from building-ledger raws.")
            return pd.DataFrame()

        result = merged.copy()
        current_year = pd.Timestamp.today().year

        text_pairs = [
            ("apt_name_ledger", "recap_apt_name_ledger", "title_apt_name_ledger"),
            ("address", "recap_address", "title_address"),
            ("sigungu_code", "recap_sigungu_code", "title_sigungu_code"),
            ("bjdong_code", "recap_bjdong_code", "title_bjdong_code"),
        ]
        for final_col, preferred_col, fallback_col in text_pairs:
            preferred = result.get(preferred_col, pd.Series(pd.NA, index=result.index, dtype="object"))
            fallback = result.get(fallback_col, pd.Series(pd.NA, index=result.index, dtype="object"))
            selected, source = self._choose_preferred(
                preferred,
                fallback,
                preferred_label="recap",
                fallback_label="title",
                valid_mask_preferred=self._valid_text(preferred),
                valid_mask_fallback=self._valid_text(fallback),
            )
            result[final_col] = selected
            result[f"{final_col}_source"] = source

        numeric_specs = {
            "completion_date": (result.get("recap_completion_date"), result.get("title_completion_date"), None),
            "land_area": (result.get("recap_land_area"), result.get("title_land_area"), (0, 10000000)),
            "floor_area_ratio_total_area": (result.get("recap_floor_area_ratio_total_area"), result.get("title_floor_area_ratio_total_area"), (0, 10000000)),
            "total_area": (result.get("recap_total_area"), result.get("title_total_area"), (0, 10000000)),
            "floor_area_ratio": (result.get("recap_floor_area_ratio"), result.get("title_floor_area_ratio"), (0, 2000)),
            "building_coverage_ratio": (result.get("recap_building_coverage_ratio"), result.get("title_building_coverage_ratio"), (0, 100)),
            "household_count": (result.get("recap_household_count"), result.get("title_household_count"), (0, 10000)),
            "ground_floor_count": (result.get("recap_ground_floor_count"), result.get("title_ground_floor_count"), (0, 250)),
            "underground_floor_count": (result.get("recap_underground_floor_count"), result.get("title_underground_floor_count"), (-1, 100)),
        }

        for final_col, (preferred, fallback, bounds) in numeric_specs.items():
            if final_col == "completion_date":
                preferred = preferred if preferred is not None else pd.Series(pd.NaT, index=result.index)
                fallback = fallback if fallback is not None else pd.Series(pd.NaT, index=result.index)
            else:
                preferred = preferred if preferred is not None else pd.Series(np.nan, index=result.index, dtype="float64")
                fallback = fallback if fallback is not None else pd.Series(np.nan, index=result.index, dtype="float64")
            if final_col == "completion_date":
                preferred_valid = preferred.notna() & preferred.dt.year.between(1900, current_year + 2)
                fallback_valid = fallback.notna() & fallback.dt.year.between(1900, current_year + 2)
            else:
                lower, upper = bounds if bounds is not None else (None, None)
                preferred_valid = self._valid_numeric(preferred, lower=lower, upper=upper)
                fallback_valid = self._valid_numeric(fallback, lower=lower, upper=upper)
            selected, source = self._choose_preferred(
                preferred,
                fallback,
                preferred_label="recap",
                fallback_label="title",
                valid_mask_preferred=preferred_valid,
                valid_mask_fallback=fallback_valid,
            )
            result[final_col] = selected if final_col == "completion_date" else pd.to_numeric(selected, errors="coerce")
            result[f"{final_col}_source"] = source

        selected_households = pd.to_numeric(result["household_count"], errors="coerce")
        recap_parking = result.get("recap_total_parking_count", pd.Series(np.nan, index=result.index, dtype="float64"))
        title_parking = result.get("title_total_parking_count", pd.Series(np.nan, index=result.index, dtype="float64"))

        recap_parking_valid = self._valid_numeric(recap_parking, lower=0, upper=30000)
        title_parking_valid = self._valid_numeric(title_parking, lower=0, upper=30000)
        parking_ratio_recap = pd.to_numeric(recap_parking, errors="coerce") / selected_households.replace(0, np.nan)
        parking_ratio_title = pd.to_numeric(title_parking, errors="coerce") / selected_households.replace(0, np.nan)
        recap_parking_valid &= parking_ratio_recap.isna() | parking_ratio_recap.le(8)
        title_parking_valid &= parking_ratio_title.isna() | parking_ratio_title.le(8)

        parking_selected, parking_source = self._choose_preferred(
            recap_parking,
            title_parking,
            preferred_label="recap",
            fallback_label="title",
            valid_mask_preferred=recap_parking_valid,
            valid_mask_fallback=title_parking_valid,
        )
        result["total_parking_count"] = pd.to_numeric(parking_selected, errors="coerce")
        result["total_parking_count_source"] = parking_source
        result["parking_value_source"] = pd.Series(pd.NA, index=result.index, dtype="string")
        recap_source = result.get("recap_parking_count_source", pd.Series(pd.NA, index=result.index, dtype="string")).astype("string")
        title_source = result.get("title_parking_count_source", pd.Series(pd.NA, index=result.index, dtype="string")).astype("string")
        recap_mask = parking_source == "recap"
        title_mask = parking_source == "title"
        result.loc[recap_mask, "parking_value_source"] = "recap_" + recap_source[recap_mask].fillna("unknown")
        result.loc[title_mask, "parking_value_source"] = "title_" + title_source[title_mask].fillna("unknown")

        safe_households = result["household_count"].replace(0, np.nan)
        result["parking_per_household"] = result["total_parking_count"] / safe_households
        result["avg_land_area_per_household"] = result["land_area"] / safe_households
        result["avg_total_area_per_household"] = result["total_area"] / safe_households
        result["completion_date"] = pd.to_datetime(result["completion_date"], errors="coerce")

        result["complex_scale_bucket"] = pd.cut(
            result["household_count"],
            bins=[0, 300, 1000, 2000, float("inf")],
            labels=["small", "medium", "large", "mega"],
            right=False,
        ).astype("string")
        result["density_bucket"] = pd.cut(
            result["floor_area_ratio"],
            bins=[0, 200, 300, 400, float("inf")],
            labels=["low", "mid", "high", "very_high"],
            right=False,
        ).astype("string")

        completion_year = result["completion_date"].dt.year
        complex_age = current_year - completion_year
        complex_age = complex_age.where(complex_age >= 0)
        far_fill = result["floor_area_ratio"]
        bcr_fill = result["building_coverage_ratio"]
        land_fill = result["avg_land_area_per_household"]
        far_component = ((250 - far_fill).clip(lower=0, upper=250) / 250.0).fillna(0) * 25.0
        bcr_component = ((35 - bcr_fill).clip(lower=0, upper=35) / 35.0).fillna(0) * 15.0
        age_component = ((complex_age - 15).clip(lower=0, upper=25) / 25.0).fillna(0) * 40.0
        land_component = ((land_fill - 20).clip(lower=0, upper=40) / 40.0).fillna(0) * 20.0
        result["redevelopment_option_score"] = (far_component + bcr_component + age_component + land_component).round(2)

        output_columns = [
            "aptSeq",
            "apt_name_ledger",
            "address",
            "sigungu_code",
            "bjdong_code",
            "completion_date",
            "land_area",
            "floor_area_ratio_total_area",
            "total_area",
            "floor_area_ratio",
            "building_coverage_ratio",
            "household_count",
            "total_parking_count",
            "parking_per_household",
            "avg_land_area_per_household",
            "avg_total_area_per_household",
            "ground_floor_count",
            "underground_floor_count",
            "complex_scale_bucket",
            "density_bucket",
            "redevelopment_option_score",
            "apt_name_ledger_source",
            "address_source",
            "sigungu_code_source",
            "bjdong_code_source",
            "completion_date_source",
            "land_area_source",
            "floor_area_ratio_total_area_source",
            "total_area_source",
            "floor_area_ratio_source",
            "building_coverage_ratio_source",
            "household_count_source",
            "ground_floor_count_source",
            "underground_floor_count_source",
            "total_parking_count_source",
            "parking_value_source",
            "title_total_parking_count_api",
            "title_parking_component_count",
            "title_total_parking_count",
            "title_parking_count_source",
            "recap_total_parking_count_api",
            "recap_parking_component_count",
            "recap_total_parking_count",
            "recap_parking_count_source",
        ]
        output_columns = [column for column in output_columns if column in result.columns]
        result = result[output_columns].sort_values("aptSeq").reset_index(drop=True)
        result.to_parquet(self.out_path, index=False)
        logger.info("Apartment lookup table saved: {} ({} rows)", self.out_path, len(result))
        return result


if __name__ == "__main__":
    BuildingLedgerSummarizer().summarize()
