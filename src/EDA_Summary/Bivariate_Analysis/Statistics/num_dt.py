"""
Numeric-Datetime Bivariate Exploratory Data Analysis.

Deterministic temporal analysis between numeric and datetime features:
- Spearman correlation with time
- Temporal aggregation
- Rolling statistics
- Highest / lowest temporal periods

The generated Excel workbook contains deterministic evidence for
downstream EDA, RAG, and SQL agents.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import BASE_DIR
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class NumericDatetimeEDA:
    """Perform deterministic bivariate EDA between numeric and datetime columns."""

    WORKBOOK_NAME = "num_datetime_eda.xlsx"
    TEMPORAL_FREQUENCY = "M"
    ROLLING_WINDOW = 7
    MIN_OBSERVATIONS = 3
    DATETIME_DETECTION_THRESHOLD = 0.80
    TOP_N_PERIODS = 5

    REPORT_DIR = BASE_DIR / "src/EDA_Summary/Bivariate_Analysis/Reports/excel_reports"
    LOG_DIR = BASE_DIR / "src/EDA_Summary/Bivariate_Analysis/logs"
    CLASSIFIED_COLUMNS_PATH = BASE_DIR / "src/Data/classified_columns.json"

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.df = df
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = get_log("NumericDatetimeEDA", log_dir=self.LOG_DIR)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @track_performance
    def load_classified_columns(self) -> dict[str, Any]:
        """Load classified column metadata from JSON."""
        if not self.CLASSIFIED_COLUMNS_PATH.exists():
            self.logger.warning(
                "classified_columns.json not found: %s", self.CLASSIFIED_COLUMNS_PATH
            )
            return {}

        try:
            with self.CLASSIFIED_COLUMNS_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise CustomException(f"Invalid classified_columns.json: {exc}") from exc
        except OSError as exc:
            raise CustomException(f"Unable to read classified columns file: {exc}") from exc

        if not isinstance(data, dict):
            raise CustomException("classified_columns.json must contain a JSON object.")
        return data

    @track_performance
    def get_numeric_columns(self) -> list[str]:
        """Resolve numeric columns using classification with dtype fallback."""
        if self.df is None:
            raise CustomException("DataFrame is required to resolve numeric columns.")

        classified = self.load_classified_columns()
        configured = classified.get("bivariate_candidates", {}).get("numeric", [])

        if configured:
            resolved = [
                col for col in configured
                if col in self.df.columns and not pd.api.types.is_bool_dtype(self.df[col])
            ]
            if resolved:
                self.logger.info("Resolved %d numeric columns from classification.", len(resolved))
                return resolved

        resolved = [
            col for col in self.df.select_dtypes(include=np.number).columns
            if not pd.api.types.is_bool_dtype(self.df[col])
        ]
        self.logger.info("Using numeric dtype fallback: %d columns.", len(resolved))
        return resolved

    @track_performance
    def get_datetime_columns(self) -> list[str]:
        """
        Resolve datetime columns.
        Classification is preferred; non-numeric columns are checked via conversion threshold.
        """
        if self.df is None:
            raise CustomException("DataFrame is required to resolve datetime columns.")

        classified = self.load_classified_columns()
        configured = classified.get("bivariate_candidates", {}).get("datetime", [])
        datetime_columns: set[str] = {col for col in configured if col in self.df.columns}

        for column in self.df.columns:
            if column in datetime_columns or pd.api.types.is_numeric_dtype(self.df[column]):
                continue
            try:
                non_null = self.df[column].dropna()
                if non_null.empty:
                    continue
                converted = pd.to_datetime(non_null, errors="coerce")
                if converted.notna().mean() >= self.DATETIME_DETECTION_THRESHOLD:
                    datetime_columns.add(column)
            except (TypeError, ValueError, OverflowError) as exc:
                self.logger.debug(
                    "Column '%s' could not be evaluated as datetime: %s", column, exc
                )

        resolved = [col for col in self.df.columns if col in datetime_columns]
        self.logger.info("Resolved %d datetime columns.", len(resolved))
        return resolved

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @track_performance
    def _validate_dataframe(self) -> None:
        if self.df is None:
            raise CustomException("DataFrame cannot be None.")
        if self.df.empty:
            raise CustomException("Cannot perform NUM-DATETIME EDA on an empty DataFrame.")
        if self.df.shape[1] < 2:
            raise CustomException("NUM-DATETIME EDA requires at least two columns.")

    @track_performance
    def _validate_frequency(self, frequency: str) -> None:
        if not isinstance(frequency, str) or not frequency.strip():
            raise CustomException("Temporal frequency must be a non-empty string.")
        try:
            pd.Period("2025-01-01", freq=frequency)
        except (ValueError, TypeError) as exc:
            raise CustomException(f"Invalid temporal frequency '{frequency}': {exc}") from exc

    @track_performance
    def _validate_window(self, window: int) -> None:
        if not isinstance(window, int):
            raise CustomException("Rolling window must be an integer.")
        if window < 2:
            raise CustomException("Rolling window must be at least 2.")

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    @track_performance
    def _convert_to_datetime(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce")

    @staticmethod
    @track_performance
    def classify_trend(correlation: float) -> str:
        """Classify trend strength and direction from Spearman correlation."""
        if pd.isna(correlation):
            return "Insufficient data"

        abs_corr = abs(correlation)
        if abs_corr >= 0.70:
            strength = "Strong"
        elif abs_corr >= 0.40:
            strength = "Moderate"
        elif abs_corr >= 0.20:
            strength = "Weak"
        else:
            strength = "Very weak"

        if abs_corr < 0.20:
            direction = "No clear trend"
        elif correlation > 0:
            direction = "Increasing"
        elif correlation < 0:
            direction = "Decreasing"
        else:
            direction = "No clear trend"

        return f"{strength} {direction}"

    @staticmethod
    def _safe_percentage_change(first: float, last: float) -> float:
        if pd.isna(first) or pd.isna(last) or first == 0:
            return np.nan
        return float(((last - first) / abs(first)) * 100)

    # ------------------------------------------------------------------
    # Pair utilities
    # ------------------------------------------------------------------

    @track_performance
    def _get_valid_pair(self, numeric_feature: str, datetime_feature: str) -> pd.DataFrame:
        if self.df is None:
            raise CustomException("DataFrame is not initialized.")

        missing = [
            col for col in (numeric_feature, datetime_feature) if col not in self.df.columns
        ]
        if missing:
            raise CustomException(f"Columns not found in DataFrame: {missing}")

        pair = self.df[[numeric_feature, datetime_feature]].copy()
        pair[numeric_feature] = (
            pd.to_numeric(pair[numeric_feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        pair[datetime_feature] = self._convert_to_datetime(pair[datetime_feature])
        return pair.dropna(subset=[numeric_feature, datetime_feature])

    @track_performance
    def _get_feature_pairs(
        self, numeric_columns: list[str], datetime_columns: list[str]
    ) -> list[tuple[str, str]]:
        return list(product(numeric_columns, datetime_columns))

    @track_performance
    def _safely_process_pairs(
        self,
        pairs: list[tuple[str, str]],
        processor: Callable[[str, str], dict[str, Any] | list[dict[str, Any]] | None],
    ) -> pd.DataFrame:
        results: list[dict[str, Any]] = []
        for numeric_feature, datetime_feature in pairs:
            try:
                result = processor(numeric_feature, datetime_feature)
                if result is None:
                    continue
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except Exception as exc:
                self.logger.warning(
                    "Error processing pair (%s, %s): %s",
                    numeric_feature, datetime_feature, exc,
                )
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Trend summary
    # ------------------------------------------------------------------

    @track_performance
    def generate_trend_summary(
        self,
        numeric_columns: list[str] | None = None,
        datetime_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate overall temporal trend statistics."""
        numeric_columns = numeric_columns if numeric_columns is not None else self.get_numeric_columns()
        datetime_columns = datetime_columns if datetime_columns is not None else self.get_datetime_columns()
        pairs = self._get_feature_pairs(numeric_columns, datetime_columns)
        self.logger.info("Generating trend summaries for %d pairs.", len(pairs))

        def process_pair(numeric_feature: str, datetime_feature: str) -> dict[str, Any] | None:
            pair = self._get_valid_pair(numeric_feature, datetime_feature)
            if len(pair) < self.MIN_OBSERVATIONS:
                return None

            pair = pair.sort_values(datetime_feature)
            time_values = (pair[datetime_feature] - pair[datetime_feature].min()).dt.total_seconds()
            numeric_values = pair[numeric_feature]

            if time_values.nunique() < 2 or numeric_values.nunique() < 2:
                correlation = p_value = np.nan
            else:
                correlation, p_value = spearmanr(time_values, numeric_values)

            date_start = pair[datetime_feature].min()
            date_end = pair[datetime_feature].max()

            return {
                "numeric_feature": numeric_feature,
                "datetime_feature": datetime_feature,
                "sample_size": int(len(pair)),
                "date_start": date_start,
                "date_end": date_end,
                "date_span_days": int((date_end - date_start).days),
                "mean": float(numeric_values.mean()),
                "median": float(numeric_values.median()),
                "std": float(numeric_values.std()),
                "min": float(numeric_values.min()),
                "max": float(numeric_values.max()),
                "spearman_time_correlation": float(correlation) if not pd.isna(correlation) else np.nan,
                "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                "trend": self.classify_trend(correlation),
            }

        result = self._safely_process_pairs(pairs, process_pair)
        if not result.empty:
            result = result.sort_values(
                by="spearman_time_correlation",
                key=lambda v: v.abs(),
                ascending=False,
            )
        self.logger.info("Trend summary generated with %d rows.", len(result))
        return result

    # ------------------------------------------------------------------
    # Temporal summary
    # ------------------------------------------------------------------

    @track_performance
    def generate_temporal_summary(
        self,
        numeric_columns: list[str] | None = None,
        datetime_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> pd.DataFrame:
        """Generate aggregated temporal statistics."""
        numeric_columns = numeric_columns if numeric_columns is not None else self.get_numeric_columns()
        datetime_columns = datetime_columns if datetime_columns is not None else self.get_datetime_columns()
        frequency = frequency if frequency is not None else self.TEMPORAL_FREQUENCY
        self._validate_frequency(frequency)

        pairs = self._get_feature_pairs(numeric_columns, datetime_columns)
        self.logger.info("Generating temporal summaries using frequency '%s'.", frequency)

        def process_pair(numeric_feature: str, datetime_feature: str) -> dict[str, Any] | None:
            pair = self._get_valid_pair(numeric_feature, datetime_feature)
            if len(pair) < self.MIN_OBSERVATIONS:
                return None

            pair = pair.sort_values(datetime_feature)
            pair["period"] = pair[datetime_feature].dt.to_period(frequency)
            grouped = pair.groupby("period", observed=True)[numeric_feature].mean().dropna()
            if len(grouped) < 2:
                return None

            first, last = float(grouped.iloc[0]), float(grouped.iloc[-1])
            return {
                "numeric_feature": numeric_feature,
                "datetime_feature": datetime_feature,
                "frequency": frequency,
                "number_of_periods": int(len(grouped)),
                "average_period_mean": float(grouped.mean()),
                "std_period_mean": float(grouped.std()),
                "minimum_period_mean": float(grouped.min()),
                "maximum_period_mean": float(grouped.max()),
                "first_period_mean": first,
                "last_period_mean": last,
                "percentage_change": self._safe_percentage_change(first, last),
            }

        result = self._safely_process_pairs(pairs, process_pair)
        if not result.empty:
            result = result.sort_values(
                "percentage_change", key=lambda v: v.abs(), ascending=False
            )
        self.logger.info("Temporal summary generated with %d rows.", len(result))
        return result

    # ------------------------------------------------------------------
    # Rolling summary
    # ------------------------------------------------------------------

    @track_performance
    def generate_rolling_summary(
        self,
        numeric_columns: list[str] | None = None,
        datetime_columns: list[str] | None = None,
        window: int | None = None,
    ) -> pd.DataFrame:
        """Generate rolling mean and standard deviation statistics."""
        numeric_columns = numeric_columns if numeric_columns is not None else self.get_numeric_columns()
        datetime_columns = datetime_columns if datetime_columns is not None else self.get_datetime_columns()
        window = window if window is not None else self.ROLLING_WINDOW
        self._validate_window(window)

        pairs = self._get_feature_pairs(numeric_columns, datetime_columns)
        self.logger.info("Generating rolling summaries using window=%d.", window)

        def process_pair(numeric_feature: str, datetime_feature: str) -> dict[str, Any] | None:
            pair = self._get_valid_pair(numeric_feature, datetime_feature)
            if len(pair) < window:
                return None

            pair = pair.sort_values(datetime_feature).reset_index(drop=True)
            values = pair[numeric_feature]
            rolling_mean = values.rolling(window=window, min_periods=window).mean().dropna()
            rolling_std = values.rolling(window=window, min_periods=window).std().dropna()
            if rolling_mean.empty:
                return None

            latest = float(values.iloc[-1])
            latest_mean = float(rolling_mean.iloc[-1])
            latest_std = float(rolling_std.iloc[-1])

            return {
                "numeric_feature": numeric_feature,
                "datetime_feature": datetime_feature,
                "window": window,
                "latest_date": pair[datetime_feature].iloc[-1],
                "latest_value": latest,
                "latest_rolling_mean": latest_mean,
                "latest_rolling_std": latest_std,
                "average_rolling_mean": float(rolling_mean.mean()),
                "minimum_rolling_mean": float(rolling_mean.min()),
                "maximum_rolling_mean": float(rolling_mean.max()),
                "average_rolling_std": float(rolling_std.mean()),
                "latest_vs_rolling_mean_pct": self._safe_percentage_change(latest_mean, latest),
            }

        result = self._safely_process_pairs(pairs, process_pair)
        if not result.empty:
            result = result.sort_values(
                "latest_vs_rolling_mean_pct", key=lambda v: v.abs(), ascending=False
            )
        self.logger.info("Rolling summary generated with %d rows.", len(result))
        return result

    # ------------------------------------------------------------------
    # Period summary
    # ------------------------------------------------------------------

    @track_performance
    def generate_period_summary(
        self,
        numeric_columns: list[str] | None = None,
        datetime_columns: list[str] | None = None,
        frequency: str | None = None,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Identify highest and lowest temporal periods."""
        numeric_columns = numeric_columns if numeric_columns is not None else self.get_numeric_columns()
        datetime_columns = datetime_columns if datetime_columns is not None else self.get_datetime_columns()
        frequency = frequency if frequency is not None else self.TEMPORAL_FREQUENCY
        top_n = top_n if top_n is not None else self.TOP_N_PERIODS
        self._validate_frequency(frequency)

        if not isinstance(top_n, int) or top_n < 1:
            raise CustomException("top_n must be a positive integer.")

        pairs = self._get_feature_pairs(numeric_columns, datetime_columns)
        self.logger.info("Generating top/bottom %d temporal periods.", top_n)

        all_results: list[pd.DataFrame] = []
        columns = [
            "numeric_feature", "datetime_feature", "period_type",
            "period", "count", "mean", "median", "std",
        ]

        for numeric_feature, datetime_feature in pairs:
            try:
                pair = self._get_valid_pair(numeric_feature, datetime_feature)
                if len(pair) < self.MIN_OBSERVATIONS:
                    continue

                pair = pair.copy()
                pair["period"] = pair[datetime_feature].dt.to_period(frequency)
                grouped = (
                    pair.groupby("period", observed=True)[numeric_feature]
                    .agg(count="count", mean="mean", median="median", std="std")
                    .reset_index()
                )
                if grouped.empty:
                    continue

                grouped["numeric_feature"] = numeric_feature
                grouped["datetime_feature"] = datetime_feature
                grouped["period"] = grouped["period"].astype(str)

                n = min(top_n, len(grouped))
                highest = grouped.nlargest(n, "mean").assign(period_type="Highest")
                lowest = grouped.nsmallest(n, "mean").assign(period_type="Lowest")
                all_results.append(pd.concat([highest, lowest], ignore_index=True))

            except Exception as exc:
                self.logger.warning(
                    "Error generating period summary for (%s, %s): %s",
                    numeric_feature, datetime_feature, exc,
                )

        if not all_results:
            return pd.DataFrame(columns=columns)
        return pd.concat(all_results, ignore_index=True)[columns]

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------

    @track_performance
    def format_excel_workbook(self, writer: pd.ExcelWriter) -> None:
        """Apply deterministic formatting to all workbook sheets."""
        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            if worksheet.max_row > 1:
                worksheet.auto_filter.ref = worksheet.dimensions

            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
                cell.alignment = cell.alignment.copy(horizontal="center")

            for column_cells in worksheet.columns:
                if not column_cells:
                    continue
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells[:100]:
                    try:
                        max_length = max(max_length, len(str(cell.value)))
                    except Exception:
                        continue
                worksheet.column_dimensions[column_letter].width = min(
                    max(max_length + 2, 10), 30
                )

    @track_performance
    def save_workbook(self, results: dict[str, pd.DataFrame]) -> Path:
        """Save all deterministic EDA results to Excel."""
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = self.REPORT_DIR / self.WORKBOOK_NAME

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for sheet_name, dataframe in results.items():
                    dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
                self.format_excel_workbook(writer)
        except OSError as exc:
            self.logger.error("Unable to write workbook: %s", exc)
            raise CustomException(f"Unable to write workbook: {exc}") from exc
        except Exception as exc:
            self.logger.error("Failed to save Excel workbook: %s", exc)
            raise CustomException(f"Failed to save NUM-DATETIME workbook: {exc}") from exc

        self.logger.info("Workbook successfully created: %s", path)
        return path

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    @track_performance
    def run_eda(self, df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
        """Execute the complete numeric-datetime EDA pipeline."""
        if df is not None:
            self.df = df

        self._validate_dataframe()
        numeric_columns = self.get_numeric_columns()
        datetime_columns = self.get_datetime_columns()

        if not numeric_columns:
            raise CustomException("No numeric columns available for NUM-DATETIME EDA.")
        if not datetime_columns:
            raise CustomException("No datetime columns available for NUM-DATETIME EDA.")

        self.logger.info("Starting NUM-DATETIME exploratory data analysis.")
        self.logger.info("Numeric columns: %s", numeric_columns)
        self.logger.info("Datetime columns: %s", datetime_columns)

        results = {
            "trend_summary": self.generate_trend_summary(numeric_columns, datetime_columns),
            "temporal_summary": self.generate_temporal_summary(
                numeric_columns, datetime_columns, frequency=self.TEMPORAL_FREQUENCY
            ),
            "rolling_summary": self.generate_rolling_summary(
                numeric_columns, datetime_columns, window=self.ROLLING_WINDOW
            ),
            "period_summary": self.generate_period_summary(
                numeric_columns, datetime_columns,
                frequency=self.TEMPORAL_FREQUENCY, top_n=self.TOP_N_PERIODS,
            ),
        }

        path = self.save_workbook(results)
        self.logger.info(
            "NUM-DATETIME EDA pipeline completed successfully. Workbook: %s", path
        )
        return results

    @track_performance
    def run_from_file(self, file_path: Path) -> dict[str, pd.DataFrame]:
        """Load a CSV dataset and execute NUM-DATETIME EDA."""
        file_path = Path(file_path)
        self.logger.info("Attempting to load dataset from: %s", file_path)

        if not file_path.exists():
            raise CustomException(f"Dataset file does not exist: {file_path}")

        try:
            df = pd.read_csv(file_path)
            self.logger.info("Dataset successfully loaded. Shape: %s", df.shape)
            return self.run_eda(df)
        except pd.errors.EmptyDataError as exc:
            raise CustomException(f"Dataset file is empty: {file_path}") from exc
        except pd.errors.ParserError as exc:
            raise CustomException(f"Unable to parse CSV file: {file_path}") from exc
        except CustomException:
            raise
        except Exception as exc:
            raise CustomException(f"Failed to execute NUM-DATETIME EDA: {exc}") from exc


@track_performance
def num_datetime_stats(file_path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Run numeric-datetime EDA from the default or supplied CSV."""
    if file_path is None:
        file_path = BASE_DIR / "src/Data/cleaned_ecommerce_dataset.csv"
    return NumericDatetimeEDA().run_from_file(Path(file_path))


if __name__ == "__main__":
    num_datetime_stats()