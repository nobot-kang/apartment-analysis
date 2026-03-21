"""Representative 59-type and 84-type complex analytics pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import ALL_REGIONS, PROCESSED_DIR


class RepresentativeComplexPipeline:
    """Build precomputed parquet datasets for representative-complex analyses."""

    DASHBOARD_START_DATE = pd.Timestamp("2020-01-01")
    DASHBOARD_START_YM = "202001"
    REPRESENTATIVE_FILL_LIMIT = 12
    AREA_BANDS: dict[int, tuple[float, float]] = {
        59: (58.0, 60.0),
        84: (83.0, 85.0),
    }
    JEONSE_LABEL = "전세"
    WOLSE_LABEL = "월세"
    STATIC_COLUMNS = [
        "aptSeq",
        "apt_name",
        "apt_name_repr",
        "dong_name",
        "dong_repr",
        "sigungu_code",
        "bjdong_code",
        "completion_date",
        "completion_year",
        "household_count",
        "total_parking_count",
        "parking_per_household",
        "floor_area_ratio",
        "building_coverage_ratio",
        "avg_land_area_per_household",
        "avg_total_area_per_household",
        "ground_floor_count",
        "underground_floor_count",
        "complex_scale_bucket",
        "density_bucket",
        "redevelopment_option_score",
        "feature_missing_count",
        "floor_area_ratio_missing",
        "building_coverage_ratio_missing",
        "avg_land_area_per_household_missing",
        "avg_total_area_per_household_missing",
        "total_parking_count_missing",
        "parking_per_household_missing",
    ]
    MACRO_COLUMNS = [
        "ym",
        "date",
        "bok_rate",
        "bok_rate_change_3m",
        "cpi_kr",
        "cpi_us",
        "fed_rate",
        "m2",
        "m2_yoy",
        "usdkrw",
    ]

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _trimmed_mean(series: pd.Series, trim_pct: float = 0.1) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna().sort_values()
        if values.empty:
            return np.nan
        trim_count = int(len(values) * trim_pct)
        if trim_count > 0 and len(values) > trim_count * 2:
            values = values.iloc[trim_count:-trim_count]
        return float(values.mean()) if not values.empty else np.nan

    @staticmethod
    def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
        valid_values = pd.to_numeric(values, errors="coerce")
        valid_weights = pd.to_numeric(weights, errors="coerce")
        mask = valid_values.notna() & valid_weights.notna()
        if mask.sum() == 0:
            return np.nan
        total_weight = float(valid_weights[mask].sum())
        if total_weight <= 0:
            return np.nan
        return float((valid_values[mask] * valid_weights[mask]).sum() / total_weight)

    @staticmethod
    def _read_parquet_optional_columns(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if columns is None:
            return pd.read_parquet(path)
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception:
            df = pd.read_parquet(path)
            available = [column for column in columns if column in df.columns]
            if not available:
                return pd.DataFrame()
            return df[available].copy()

    def _load_processed_chunks(self, prefix: str, columns: list[str] | None = None) -> pd.DataFrame:
        files = sorted(self.output_dir.glob(f"{prefix}_*.parquet"))
        if not files:
            full_path = self.output_dir / f"{prefix}.parquet"
            if not full_path.exists():
                return pd.DataFrame()
            return self._read_parquet_optional_columns(full_path, columns)

        frames = [self._read_parquet_optional_columns(path, columns) for path in files]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _write_output_parquet(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty and len(df.columns) == 0:
            logger.warning("Representative dataset is empty: {}", name)
            return df
        df.to_parquet(self.output_dir / name, index=False)
        return df

    def _filter_dashboard_window(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        if df.empty:
            return df.copy()
        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
        return result[result[date_col] >= self.DASHBOARD_START_DATE].copy()

    def _load_complex_master(self) -> pd.DataFrame:
        path = self.output_dir / "complex_master.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _load_macro_monthly(self) -> pd.DataFrame:
        path = self.output_dir / "macro_monthly.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "ym" not in df.columns:
            df["ym"] = df["date"].dt.strftime("%Y%m")
        df = df[df["ym"] >= self.DASHBOARD_START_YM].copy()
        if "bok_rate_change_3m" not in df.columns and "bok_rate" in df.columns:
            df["bok_rate_change_3m"] = pd.to_numeric(df["bok_rate"], errors="coerce").diff(3)
        if "m2_yoy" not in df.columns and "m2" in df.columns:
            df["m2_yoy"] = pd.to_numeric(df["m2"], errors="coerce").pct_change(12) * 100
        available = [column for column in self.MACRO_COLUMNS if column in df.columns]
        return df[available].sort_values("date").reset_index(drop=True)

    def _assign_area_band(self, area: pd.Series) -> pd.Series:
        numeric_area = pd.to_numeric(area, errors="coerce")
        result = pd.Series(pd.NA, index=numeric_area.index, dtype="Int64")
        for band, (low, high) in self.AREA_BANDS.items():
            result.loc[numeric_area.ge(low) & numeric_area.lt(high)] = band
        return result

    def _build_month_calendar(self, max_date: pd.Timestamp | None) -> pd.DataFrame:
        if pd.isna(max_date):
            return pd.DataFrame(columns=["ym", "date"])
        month_end = pd.Timestamp(max_date).to_period("M").to_timestamp()
        months = pd.date_range(start=self.DASHBOARD_START_DATE, end=month_end, freq="MS")
        return pd.DataFrame({"date": months, "ym": months.strftime("%Y%m")})

    def _expand_monthly_panel(
        self,
        observed_df: pd.DataFrame,
        group_cols: list[str],
        *,
        count_col: str,
        fill_map: dict[str, str],
        months_since_col: str,
        imputed_col: str,
    ) -> pd.DataFrame:
        if observed_df.empty:
            return pd.DataFrame()

        observed = observed_df.copy()
        observed["date"] = pd.to_datetime(observed["date"], errors="coerce")
        month_calendar = self._build_month_calendar(observed["date"].max())
        combos = observed[group_cols].drop_duplicates().copy()
        combos["_merge_key"] = 1
        month_calendar = month_calendar.copy()
        month_calendar["_merge_key"] = 1
        expanded = combos.merge(month_calendar, on="_merge_key", how="inner").drop(columns="_merge_key")
        expanded = expanded.merge(observed, on=[*group_cols, "ym", "date"], how="left")
        expanded[count_col] = pd.to_numeric(expanded[count_col], errors="coerce").fillna(0).astype(int)
        expanded = expanded.sort_values([*group_cols, "date"]).reset_index(drop=True)

        frames: list[pd.DataFrame] = []
        for _, group in expanded.groupby(group_cols, observed=True, sort=False):
            working = group.sort_values("date").copy()
            obs_mask = working[count_col].gt(0)
            obs_positions = pd.Series(
                np.where(obs_mask.to_numpy(), np.arange(len(working)), np.nan),
                index=working.index,
                dtype="float64",
            ).ffill()
            months_since = pd.Series(pd.NA, index=working.index, dtype="Int64")
            if obs_positions.notna().any():
                valid_idx = obs_positions.notna()
                distances = np.arange(len(working), dtype="int64")[valid_idx.to_numpy()] - obs_positions[valid_idx].astype(int).to_numpy()
                months_since.loc[valid_idx] = distances.astype("int64")
            fill_active = months_since.notna() & months_since.le(self.REPRESENTATIVE_FILL_LIMIT)
            working[months_since_col] = months_since.where(fill_active, pd.NA).astype("Int64")
            working["fill_active"] = fill_active.astype(bool)
            working[imputed_col] = working["fill_active"] & ~obs_mask
            for observed_col, filled_col in fill_map.items():
                numeric_values = pd.to_numeric(working[observed_col], errors="coerce")
                working[filled_col] = numeric_values.ffill(limit=self.REPRESENTATIVE_FILL_LIMIT)
            frames.append(working)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _prepare_representative_trade_observed(self) -> pd.DataFrame:
        trade = self._load_processed_chunks(
            "apt_trade",
            columns=["date", "aptSeq", "price", "price_per_py", "area"],
        )
        trade = self._filter_dashboard_window(trade, date_col="date")
        if trade.empty:
            return pd.DataFrame()

        trade = trade.copy()
        trade["date"] = pd.to_datetime(trade["date"], errors="coerce")
        trade["aptSeq"] = trade["aptSeq"].astype("string")
        trade["area_band"] = self._assign_area_band(trade["area"])
        trade = trade.dropna(subset=["aptSeq", "date", "area_band", "price_per_py"]).copy()
        trade["ym"] = trade["date"].dt.strftime("%Y%m")

        group_cols = ["aptSeq", "ym", "area_band"]
        summary = (
            trade.groupby(group_cols, observed=True)
            .agg(
                trade_count_obs=("price_per_py", "size"),
                price_per_py_obs_median=("price_per_py", "median"),
                price_per_py_obs_mean=("price_per_py", "mean"),
                price_mean_obs=("price", "mean"),
                area_median_obs=("area", "median"),
            )
            .reset_index()
        )
        trimmed = (
            trade.groupby(group_cols, observed=True)["price_per_py"]
            .apply(self._trimmed_mean)
            .reset_index(name="price_per_py_obs_trimmed")
        )
        summary = summary.merge(trimmed, on=group_cols, how="left")
        summary["date"] = pd.to_datetime(summary["ym"], format="%Y%m", errors="coerce")
        summary["area_band"] = summary["area_band"].astype("Int64")
        return summary.sort_values(["aptSeq", "date", "area_band"]).reset_index(drop=True)

    def _prepare_representative_rent_observed(self) -> pd.DataFrame:
        rent = self._load_processed_chunks(
            "apt_rent",
            columns=[
                "date",
                "aptSeq",
                "area",
                "rentType",
                "deposit_per_py",
                "monthly_rent_per_py",
                "monthly_rent",
            ],
        )
        rent = self._filter_dashboard_window(rent, date_col="date")
        if rent.empty:
            return pd.DataFrame()

        rent = rent.copy()
        rent["date"] = pd.to_datetime(rent["date"], errors="coerce")
        rent["aptSeq"] = rent["aptSeq"].astype("string")
        if "rentType" not in rent.columns:
            rent["rentType"] = np.where(
                pd.to_numeric(rent["monthly_rent"], errors="coerce").fillna(0) == 0,
                self.JEONSE_LABEL,
                self.WOLSE_LABEL,
            )
        rent["area_band"] = self._assign_area_band(rent["area"])
        rent = rent.dropna(subset=["aptSeq", "date", "area_band", "rentType"]).copy()
        rent["ym"] = rent["date"].dt.strftime("%Y%m")

        group_cols = ["aptSeq", "ym", "area_band", "rentType"]
        summary = (
            rent.groupby(group_cols, observed=True)
            .agg(
                rent_count_obs=("rentType", "size"),
                deposit_per_py_obs_median=("deposit_per_py", "median"),
                deposit_per_py_obs_mean=("deposit_per_py", "mean"),
                monthly_rent_per_py_obs_median=("monthly_rent_per_py", "median"),
                monthly_rent_per_py_obs_mean=("monthly_rent_per_py", "mean"),
            )
            .reset_index()
        )
        summary["date"] = pd.to_datetime(summary["ym"], format="%Y%m", errors="coerce")
        summary["area_band"] = summary["area_band"].astype("Int64")
        return summary.sort_values(["aptSeq", "date", "area_band", "rentType"]).reset_index(drop=True)

    def build_representative_complex_universe(
        self,
        *,
        trade_observed: pd.DataFrame | None = None,
        rent_observed: pd.DataFrame | None = None,
        complex_master: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        trade_observed = trade_observed if trade_observed is not None else self._prepare_representative_trade_observed()
        rent_observed = rent_observed if rent_observed is not None else self._prepare_representative_rent_observed()
        complex_master = complex_master if complex_master is not None else self._load_complex_master()

        base_columns = [column for column in self.STATIC_COLUMNS if column in complex_master.columns]
        base = complex_master[base_columns].copy() if not complex_master.empty else pd.DataFrame(columns=["aptSeq"])
        if "aptSeq" in base.columns:
            base["aptSeq"] = base["aptSeq"].astype("string")
        if "sigungu_code" in base.columns:
            base["sigungu_name"] = base["sigungu_code"].astype("string").map(ALL_REGIONS).fillna(base["sigungu_code"])
        else:
            base["sigungu_name"] = pd.NA

        def _pivot_month_counts(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(columns=["aptSeq"])
            return (
                df.groupby(["aptSeq", "area_band"], observed=True)["ym"]
                .nunique()
                .unstack("area_band")
                .rename(columns={59: f"{prefix}_months_59", 84: f"{prefix}_months_84"})
                .reset_index()
            )

        def _bounds(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(columns=["aptSeq"])
            return (
                df.groupby("aptSeq", observed=True)["ym"]
                .agg(**{f"first_{prefix}_ym": "min", f"last_{prefix}_ym": "max"})
                .reset_index()
            )

        trade_counts = _pivot_month_counts(trade_observed, "trade")
        rent_counts = _pivot_month_counts(rent_observed, "rent")
        trade_bounds = _bounds(trade_observed, "trade")
        rent_bounds = _bounds(rent_observed, "rent")

        universe = base.merge(trade_counts, on="aptSeq", how="outer")
        universe = universe.merge(rent_counts, on="aptSeq", how="outer")
        universe = universe.merge(trade_bounds, on="aptSeq", how="outer")
        universe = universe.merge(rent_bounds, on="aptSeq", how="outer")

        month_columns = [
            "trade_months_59",
            "trade_months_84",
            "rent_months_59",
            "rent_months_84",
        ]
        for column in month_columns:
            if column not in universe.columns:
                universe[column] = 0
            universe[column] = pd.to_numeric(universe[column], errors="coerce").fillna(0).astype(int)

        universe["has_trade_59"] = universe["trade_months_59"].gt(0)
        universe["has_trade_84"] = universe["trade_months_84"].gt(0)
        universe["has_rent_59"] = universe["rent_months_59"].gt(0)
        universe["has_rent_84"] = universe["rent_months_84"].gt(0)
        universe["has_59_any"] = universe["has_trade_59"] | universe["has_rent_59"]
        universe["has_84_any"] = universe["has_trade_84"] | universe["has_rent_84"]
        universe["is_trade_pair_complex"] = universe["has_trade_59"] & universe["has_trade_84"]
        universe["is_rent_pair_complex"] = universe["has_rent_59"] & universe["has_rent_84"]
        universe["is_pair_complex"] = universe["has_59_any"] & universe["has_84_any"]

        observed_bounds = pd.concat(
            [
                trade_observed[["aptSeq", "ym"]] if not trade_observed.empty else pd.DataFrame(columns=["aptSeq", "ym"]),
                rent_observed[["aptSeq", "ym"]] if not rent_observed.empty else pd.DataFrame(columns=["aptSeq", "ym"]),
            ],
            ignore_index=True,
        )
        if not observed_bounds.empty:
            overall_bounds = (
                observed_bounds.groupby("aptSeq", observed=True)["ym"]
                .agg(first_obs_ym="min", last_obs_ym="max")
                .reset_index()
            )
            universe = universe.merge(overall_bounds, on="aptSeq", how="left")

        universe = universe[universe["has_59_any"] | universe["has_84_any"]].copy()
        universe = universe.sort_values(["sigungu_code", "dong_repr", "apt_name_repr"], na_position="last").reset_index(drop=True)
        return self._write_output_parquet("representative_complex_universe.parquet", universe)

    def build_representative_trade_band_monthly(
        self,
        *,
        trade_observed: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        trade_observed = trade_observed if trade_observed is not None else self._prepare_representative_trade_observed()
        universe = universe if universe is not None else self.build_representative_complex_universe(trade_observed=trade_observed)
        if trade_observed.empty:
            return self._write_output_parquet("representative_trade_band_monthly.parquet", pd.DataFrame())

        panel = self._expand_monthly_panel(
            trade_observed,
            ["aptSeq", "area_band"],
            count_col="trade_count_obs",
            fill_map={"price_per_py_obs_median": "price_per_py_filled"},
            months_since_col="months_since_trade_obs",
            imputed_col="is_trade_imputed",
        )
        static_columns = [
            column
            for column in universe.columns
            if column not in {"first_trade_ym", "last_trade_ym", "first_rent_ym", "last_rent_ym", "first_obs_ym", "last_obs_ym"}
        ]
        panel = panel.merge(universe[static_columns].drop_duplicates(subset=["aptSeq"]), on="aptSeq", how="left")
        panel["year"] = panel["date"].dt.year
        panel["month"] = panel["date"].dt.month
        panel = panel.sort_values(["aptSeq", "area_band", "date"]).reset_index(drop=True)
        return self._write_output_parquet("representative_trade_band_monthly.parquet", panel)

    def build_representative_rent_band_monthly(
        self,
        *,
        rent_observed: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        rent_observed = rent_observed if rent_observed is not None else self._prepare_representative_rent_observed()
        universe = universe if universe is not None else self.build_representative_complex_universe(rent_observed=rent_observed)
        if rent_observed.empty:
            return self._write_output_parquet("representative_rent_band_monthly.parquet", pd.DataFrame())

        panel = self._expand_monthly_panel(
            rent_observed,
            ["aptSeq", "area_band", "rentType"],
            count_col="rent_count_obs",
            fill_map={
                "deposit_per_py_obs_median": "deposit_per_py_filled",
                "monthly_rent_per_py_obs_median": "monthly_rent_per_py_filled",
            },
            months_since_col="months_since_rent_obs",
            imputed_col="is_rent_imputed",
        )
        static_columns = [
            column
            for column in universe.columns
            if column not in {"first_trade_ym", "last_trade_ym", "first_rent_ym", "last_rent_ym", "first_obs_ym", "last_obs_ym"}
        ]
        panel = panel.merge(universe[static_columns].drop_duplicates(subset=["aptSeq"]), on="aptSeq", how="left")
        panel["year"] = panel["date"].dt.year
        panel["month"] = panel["date"].dt.month
        panel = panel.sort_values(["aptSeq", "area_band", "rentType", "date"]).reset_index(drop=True)
        return self._write_output_parquet("representative_rent_band_monthly.parquet", panel)

    @staticmethod
    def _rolling_current_median(series: pd.Series, window: int = 3) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors="coerce")
        rolled = numeric_series.rolling(window=window, min_periods=1).median()
        return rolled.where(numeric_series.notna())

    def build_representative_pair_gap_monthly(
        self,
        *,
        trade_panel: pd.DataFrame | None = None,
        rent_panel: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        trade_panel = trade_panel if trade_panel is not None else self._read_parquet_optional_columns(self.output_dir / "representative_trade_band_monthly.parquet")
        rent_panel = rent_panel if rent_panel is not None else self._read_parquet_optional_columns(self.output_dir / "representative_rent_band_monthly.parquet")
        universe = universe if universe is not None else self._read_parquet_optional_columns(self.output_dir / "representative_complex_universe.parquet")
        if universe.empty:
            return self._write_output_parquet("representative_pair_gap_monthly.parquet", pd.DataFrame())

        pair_universe = universe[universe.get("is_pair_complex", False)].copy()
        if pair_universe.empty:
            return self._write_output_parquet("representative_pair_gap_monthly.parquet", pd.DataFrame())

        max_dates = []
        for frame in (trade_panel, rent_panel):
            if not frame.empty and "date" in frame.columns:
                max_dates.append(pd.to_datetime(frame["date"], errors="coerce").max())
        month_calendar = self._build_month_calendar(max(max_dates) if max_dates else pd.NaT)
        if month_calendar.empty:
            return self._write_output_parquet("representative_pair_gap_monthly.parquet", pd.DataFrame())

        base_columns = [column for column in pair_universe.columns if column != "aptSeq"]
        base = pair_universe[["aptSeq", *base_columns]].drop_duplicates(subset=["aptSeq"]).copy()
        base["_merge_key"] = 1
        month_calendar = month_calendar.copy()
        month_calendar["_merge_key"] = 1
        pair_gap = base.merge(month_calendar, on="_merge_key", how="inner").drop(columns="_merge_key")

        def _merge_band(
            frame: pd.DataFrame,
            *,
            band: int,
            rename_map: dict[str, str],
            extra_filters: dict[str, object] | None = None,
        ) -> None:
            nonlocal pair_gap
            if frame.empty:
                return
            subset = frame.copy()
            subset["area_band"] = subset["area_band"].astype("Int64")
            subset = subset[subset["area_band"].eq(band)].copy()
            if extra_filters:
                for column, value in extra_filters.items():
                    subset = subset[subset[column] == value].copy()
            keep_cols = ["aptSeq", "ym", *rename_map.keys()]
            available_cols = [column for column in keep_cols if column in subset.columns]
            subset = subset[available_cols].rename(columns=rename_map)
            pair_gap = pair_gap.merge(subset, on=["aptSeq", "ym"], how="left")

        _merge_band(
            trade_panel,
            band=59,
            rename_map={
                "price_per_py_filled": "sale_py_59",
                "months_since_trade_obs": "sale_59_fill_age",
                "is_trade_imputed": "sale_59_imputed",
                "trade_count_obs": "sale_trade_count_59_obs",
            },
        )
        _merge_band(
            trade_panel,
            band=84,
            rename_map={
                "price_per_py_filled": "sale_py_84",
                "months_since_trade_obs": "sale_84_fill_age",
                "is_trade_imputed": "sale_84_imputed",
                "trade_count_obs": "sale_trade_count_84_obs",
            },
        )
        _merge_band(
            rent_panel,
            band=59,
            extra_filters={"rentType": self.JEONSE_LABEL},
            rename_map={
                "deposit_per_py_filled": "jeonse_py_59",
                "months_since_rent_obs": "jeonse_59_fill_age",
                "is_rent_imputed": "jeonse_59_imputed",
                "rent_count_obs": "jeonse_count_59_obs",
            },
        )
        _merge_band(
            rent_panel,
            band=84,
            extra_filters={"rentType": self.JEONSE_LABEL},
            rename_map={
                "deposit_per_py_filled": "jeonse_py_84",
                "months_since_rent_obs": "jeonse_84_fill_age",
                "is_rent_imputed": "jeonse_84_imputed",
                "rent_count_obs": "jeonse_count_84_obs",
            },
        )
        _merge_band(
            rent_panel,
            band=59,
            extra_filters={"rentType": self.WOLSE_LABEL},
            rename_map={
                "monthly_rent_per_py_filled": "wolse_py_59",
                "months_since_rent_obs": "wolse_59_fill_age",
                "is_rent_imputed": "wolse_59_imputed",
                "rent_count_obs": "wolse_count_59_obs",
            },
        )
        _merge_band(
            rent_panel,
            band=84,
            extra_filters={"rentType": self.WOLSE_LABEL},
            rename_map={
                "monthly_rent_per_py_filled": "wolse_py_84",
                "months_since_rent_obs": "wolse_84_fill_age",
                "is_rent_imputed": "wolse_84_imputed",
                "rent_count_obs": "wolse_count_84_obs",
            },
        )

        pair_gap = pair_gap.sort_values(["aptSeq", "date"]).reset_index(drop=True)
        for column in [
            "sale_py_59",
            "sale_py_84",
            "jeonse_py_59",
            "jeonse_py_84",
            "wolse_py_59",
            "wolse_py_84",
        ]:
            if column in pair_gap.columns:
                pair_gap[f"{column}_roll3"] = (
                    pair_gap.groupby("aptSeq", observed=True)[column]
                    .transform(self._rolling_current_median)
                )

        def _compute_gap(prefix: str) -> None:
            left = f"{prefix}_59_roll3"
            right = f"{prefix}_84_roll3"
            if left not in pair_gap.columns or right not in pair_gap.columns:
                return
            left_values = pd.to_numeric(pair_gap[left], errors="coerce")
            right_values = pd.to_numeric(pair_gap[right], errors="coerce")
            valid = left_values.gt(0) & right_values.gt(0)
            pair_gap[f"{prefix}_gap_abs"] = right_values - left_values
            pair_gap[f"{prefix}_gap_ratio"] = np.where(valid, (right_values / left_values - 1) * 100, np.nan)
            pair_gap[f"{prefix}_gap_log"] = np.where(valid, np.log(right_values) - np.log(left_values), np.nan)

        _compute_gap("sale_py")
        _compute_gap("jeonse_py")
        _compute_gap("wolse_py")

        rename_pairs = {
            "sale_py_gap_abs": "sale_gap_abs",
            "sale_py_gap_ratio": "sale_gap_ratio",
            "sale_py_gap_log": "sale_gap_log",
            "jeonse_py_gap_abs": "jeonse_gap_abs",
            "jeonse_py_gap_ratio": "jeonse_gap_ratio",
            "jeonse_py_gap_log": "jeonse_gap_log",
            "wolse_py_gap_abs": "wolse_gap_abs",
            "wolse_py_gap_ratio": "wolse_gap_ratio",
            "wolse_py_gap_log": "wolse_gap_log",
        }
        for source_col, target_col in rename_pairs.items():
            if source_col in pair_gap.columns:
                pair_gap[target_col] = pair_gap[source_col]

        def _bool_series(column: str) -> pd.Series:
            if column not in pair_gap.columns:
                return pd.Series(False, index=pair_gap.index, dtype=bool)
            return pair_gap[column].astype("boolean").fillna(False).astype(bool)

        pair_gap["sale_any_imputed"] = _bool_series("sale_59_imputed") | _bool_series("sale_84_imputed")
        pair_gap["jeonse_any_imputed"] = _bool_series("jeonse_59_imputed") | _bool_series("jeonse_84_imputed")
        pair_gap["wolse_any_imputed"] = _bool_series("wolse_59_imputed") | _bool_series("wolse_84_imputed")
        pair_gap["year"] = pair_gap["date"].dt.year
        pair_gap["month"] = pair_gap["date"].dt.month
        return self._write_output_parquet("representative_pair_gap_monthly.parquet", pair_gap)

    def _build_region_rows(
        self,
        frame: pd.DataFrame,
        *,
        market_type: str,
        value_col: str,
        count_col: str,
    ) -> list[dict[str, object]]:
        if frame.empty:
            return []

        rows: list[dict[str, object]] = []
        region_specs = [
            ("sigungu", "sigungu_code", "sigungu_name"),
            ("bjdong", "dong_repr", "dong_repr"),
        ]
        for region_level, code_col, name_col in region_specs:
            required_cols = {"aptSeq", "ym", "date", "area_band", value_col, count_col, code_col, name_col}
            if not required_cols.issubset(frame.columns):
                continue
            working = frame[list(required_cols)].copy()
            working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
            working[count_col] = pd.to_numeric(working[count_col], errors="coerce").fillna(0)
            working = working.dropna(subset=[value_col, code_col, name_col]).copy()
            if working.empty:
                continue
            grouped = working.groupby([code_col, name_col, "ym", "date", "area_band"], observed=True, sort=True)
            for key, group in grouped:
                region_code, region_name, ym, date, area_band = key
                values = group[value_col]
                rows.append(
                    {
                        "region_level": region_level,
                        "region_code": str(region_code),
                        "region_name": str(region_name),
                        "ym": str(ym),
                        "date": pd.to_datetime(date, errors="coerce"),
                        "area_band": int(area_band),
                        "market_type": market_type,
                        "complex_count_active": int(group["aptSeq"].nunique()),
                        "complex_count_observed": int(group.loc[group[count_col].gt(0), "aptSeq"].nunique()),
                        "complex_eq_median_py": float(values.median()),
                        "complex_eq_mean_py": float(values.mean()),
                        "complex_eq_trimmed_py": self._trimmed_mean(values),
                        "tx_weighted_mean_py": self._weighted_average(values, group[count_col]),
                        "p10": float(values.quantile(0.10)),
                        "p25": float(values.quantile(0.25)),
                        "p75": float(values.quantile(0.75)),
                        "p90": float(values.quantile(0.90)),
                        "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                        "transaction_count_obs": int(group[count_col].sum()),
                    }
                )
        return rows

    def build_representative_region_monthly(
        self,
        *,
        trade_panel: pd.DataFrame | None = None,
        rent_panel: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        trade_panel = trade_panel if trade_panel is not None else self._read_parquet_optional_columns(self.output_dir / "representative_trade_band_monthly.parquet")
        rent_panel = rent_panel if rent_panel is not None else self._read_parquet_optional_columns(self.output_dir / "representative_rent_band_monthly.parquet")

        rows: list[dict[str, object]] = []
        if not trade_panel.empty and "price_per_py_filled" in trade_panel.columns:
            rows.extend(
                self._build_region_rows(
                    trade_panel[trade_panel["price_per_py_filled"].notna()].copy(),
                    market_type="sale",
                    value_col="price_per_py_filled",
                    count_col="trade_count_obs",
                )
            )
        if not rent_panel.empty:
            jeonse = rent_panel[rent_panel["rentType"] == self.JEONSE_LABEL].copy()
            wolse = rent_panel[rent_panel["rentType"] == self.WOLSE_LABEL].copy()
            if "deposit_per_py_filled" in jeonse.columns:
                rows.extend(
                    self._build_region_rows(
                        jeonse[jeonse["deposit_per_py_filled"].notna()].copy(),
                        market_type="jeonse",
                        value_col="deposit_per_py_filled",
                        count_col="rent_count_obs",
                    )
                )
            if "monthly_rent_per_py_filled" in wolse.columns:
                rows.extend(
                    self._build_region_rows(
                        wolse[wolse["monthly_rent_per_py_filled"].notna()].copy(),
                        market_type="wolse",
                        value_col="monthly_rent_per_py_filled",
                        count_col="rent_count_obs",
                    )
                )

        region_monthly = pd.DataFrame(rows)
        if not region_monthly.empty:
            region_monthly = region_monthly.sort_values(
                ["region_level", "market_type", "region_code", "area_band", "date"]
            ).reset_index(drop=True)
        return self._write_output_parquet("representative_region_monthly.parquet", region_monthly)

    def build_representative_forecast_targets(
        self,
        *,
        pair_gap: pd.DataFrame | None = None,
        macro_monthly: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        pair_gap = pair_gap if pair_gap is not None else self._read_parquet_optional_columns(self.output_dir / "representative_pair_gap_monthly.parquet")
        macro_monthly = macro_monthly if macro_monthly is not None else self._load_macro_monthly()
        if pair_gap.empty:
            return self._write_output_parquet("representative_forecast_targets.parquet", pd.DataFrame())

        forecast = pair_gap.copy()
        forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
        if not macro_monthly.empty:
            forecast = forecast.merge(macro_monthly, on=["ym", "date"], how="left")

        grouped = forecast.groupby("aptSeq", observed=True)
        base_cols = [
            "sale_py_59_roll3",
            "sale_py_84_roll3",
            "sale_gap_ratio",
            "sale_gap_abs",
            "jeonse_gap_ratio",
            "wolse_gap_ratio",
        ]
        base_cols = [column for column in base_cols if column in forecast.columns]
        for column in base_cols:
            for lag in (1, 3, 6, 12):
                forecast[f"{column}_lag{lag}"] = grouped[column].shift(lag)

        if "sale_gap_ratio" in forecast.columns:
            forecast["sale_gap_ratio_change_1m"] = grouped["sale_gap_ratio"].diff(1)
            forecast["sale_gap_ratio_change_3m"] = grouped["sale_gap_ratio"].diff(3)
            forecast["sale_gap_ratio_change_12m"] = grouped["sale_gap_ratio"].diff(12)

        for horizon in (1, 3, 12):
            if "sale_py_59_roll3" in forecast.columns:
                future_59 = grouped["sale_py_59_roll3"].shift(-horizon)
                forecast[f"sale_py_59_t{horizon}"] = future_59
                forecast[f"future_sale_py_59_return_{horizon}m"] = (future_59 / forecast["sale_py_59_roll3"] - 1) * 100
            if "sale_py_84_roll3" in forecast.columns:
                future_84 = grouped["sale_py_84_roll3"].shift(-horizon)
                forecast[f"sale_py_84_t{horizon}"] = future_84
                forecast[f"future_sale_py_84_return_{horizon}m"] = (future_84 / forecast["sale_py_84_roll3"] - 1) * 100
            if "sale_gap_ratio" in forecast.columns:
                future_gap = grouped["sale_gap_ratio"].shift(-horizon)
                forecast[f"sale_gap_ratio_t{horizon}"] = future_gap
                forecast[f"future_sale_gap_ratio_change_{horizon}m"] = future_gap - forecast["sale_gap_ratio"]

        forecast = forecast.sort_values(["aptSeq", "date"]).reset_index(drop=True)
        return self._write_output_parquet("representative_forecast_targets.parquet", forecast)

    def run_all(
        self,
        *,
        complex_master: pd.DataFrame | None = None,
        macro_monthly: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        complex_master = complex_master if complex_master is not None else self._load_complex_master()
        macro_monthly = macro_monthly if macro_monthly is not None else self._load_macro_monthly()

        trade_observed = self._prepare_representative_trade_observed()
        rent_observed = self._prepare_representative_rent_observed()
        universe = self.build_representative_complex_universe(
            trade_observed=trade_observed,
            rent_observed=rent_observed,
            complex_master=complex_master,
        )
        trade_panel = self.build_representative_trade_band_monthly(
            trade_observed=trade_observed,
            universe=universe,
        )
        rent_panel = self.build_representative_rent_band_monthly(
            rent_observed=rent_observed,
            universe=universe,
        )
        pair_gap = self.build_representative_pair_gap_monthly(
            trade_panel=trade_panel,
            rent_panel=rent_panel,
            universe=universe,
        )
        region_monthly = self.build_representative_region_monthly(
            trade_panel=trade_panel,
            rent_panel=rent_panel,
        )
        forecast_targets = self.build_representative_forecast_targets(
            pair_gap=pair_gap,
            macro_monthly=macro_monthly,
        )
        logger.info(
            "Representative datasets saved: universe={}, trade_panel={}, rent_panel={}, pair_gap={}, region_monthly={}, forecast_targets={}",
            len(universe),
            len(trade_panel),
            len(rent_panel),
            len(pair_gap),
            len(region_monthly),
            len(forecast_targets),
        )
        return {
            "universe": universe,
            "trade_panel": trade_panel,
            "rent_panel": rent_panel,
            "pair_gap": pair_gap,
            "region_monthly": region_monthly,
            "forecast_targets": forecast_targets,
        }
