"""
Categorical-Datetime Bivariate Exploratory Data Analysis.
Deterministic temporal analysis between categorical and datetime features:
- Category frequency over time
- Category proportion over time
- Chi-square association across periods
- Cramér's V
- Category proportion trends
Excel workbook provides deterministic evidence for downstream EDA/RAG/SQL agents.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.config import BASE_DIR
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class CategoricalDatetimeEDA:
    """Bivariate EDA between categorical and datetime variables."""

    WORKBOOK_NAME = "cat_datetime_eda.xlsx"
    TEMPORAL_FREQUENCY = "M"
    MIN_CATEGORY_COUNT = 5
    TOP_N_CATEGORIES = 10
    MIN_OBSERVATIONS = 5
    MIN_PERIODS = 3
    DATETIME_DETECTION_THRESHOLD = 0.80
    REPORT_DIR = BASE_DIR / "src" / "EDA_Summary" / "Bivariate_Analysis" / "Reports" / "excel_reports"
    LOG_DIR = BASE_DIR / "src" / "EDA_Summary" / "Bivariate_Analysis" / "logs"
    CLASSIFIED_COLUMNS_PATH = BASE_DIR / "src" / "Data" / "classified_columns.json"

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.df = df
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = get_log("CategoricalDatetimeEDA", log_dir=self.LOG_DIR)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    @track_performance
    def load_classified_columns(self) -> dict[str, Any]:
        if not self.CLASSIFIED_COLUMNS_PATH.exists():
            self.logger.warning("classified_columns.json not found: %s", self.CLASSIFIED_COLUMNS_PATH)
            return {}
        try:
            with self.CLASSIFIED_COLUMNS_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise CustomException(f"Invalid classified_columns.json: {exc}") from exc
        except OSError as exc:
            raise CustomException(f"Unable to read classified columns file: {exc}") from exc

    @track_performance
    def get_categorical_columns(self) -> list[str]:
        if self.df is None:
            raise CustomException("DataFrame is required to resolve categorical columns.")
        configured = (
            self.load_classified_columns()
            .get("bivariate_candidates", {})
            .get("categorical", [])
        )
        if configured:
            cols = [c for c in configured if c in self.df.columns]
            if cols:
                self.logger.info("Resolved %d categorical columns from classification.", len(cols))
                return cols
        cols = self.df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        self.logger.info("Using categorical dtype fallback: %d columns.", len(cols))
        return cols

    @track_performance
    def get_datetime_columns(self) -> list[str]:
        if self.df is None:
            raise CustomException("DataFrame is required to resolve datetime columns.")
        configured = (
            self.load_classified_columns()
            .get("bivariate_candidates", {})
            .get("datetime", [])
        )
        dt_cols: set[str] = {c for c in configured if c in self.df.columns}
        for col in self.df.columns:
            if col in dt_cols or pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            try:
                non_null = self.df[col].dropna()
                if non_null.empty:
                    continue
                ratio = pd.to_datetime(non_null, errors="coerce").notna().mean()
                if ratio >= self.DATETIME_DETECTION_THRESHOLD:
                    dt_cols.add(col)
            except Exception as exc:
                self.logger.warning("Unable to evaluate datetime column '%s': %s", col, exc)
        resolved = [c for c in self.df.columns if c in dt_cols]
        self.logger.info("Resolved %d datetime columns.", len(resolved))
        return resolved

    # ------------------------------------------------------------------
    # Validation & utilities
    # ------------------------------------------------------------------
    @track_performance
    def _validate_dataframe(self) -> None:
        if self.df is None:
            raise CustomException("DataFrame cannot be None.")
        if self.df.empty:
            raise CustomException("Cannot perform CAT-DATETIME EDA on an empty DataFrame.")
        if self.df.shape[1] < 2:
            raise CustomException("CAT-DATETIME EDA requires at least two columns.")

    @staticmethod
    @track_performance
    def _convert_to_datetime(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce")

    @staticmethod
    @track_performance
    def _clean_category(series: pd.Series) -> pd.Series:
        cleaned = (
            series.astype("object")
            .where(series.notna(), "Missing")
            .astype(str)
            .str.strip()
        )
        return cleaned.replace({"": "Missing", "nan": "Missing", "None": "Missing", "NaN": "Missing"})

    @track_performance
    def _get_valid_pair(self, cat_feature: str, dt_feature: str) -> pd.DataFrame:
        if self.df is None:
            raise CustomException("DataFrame is not initialized.")
        pair = self.df[[cat_feature, dt_feature]].copy()
        pair[cat_feature] = self._clean_category(pair[cat_feature])
        pair[dt_feature] = self._convert_to_datetime(pair[dt_feature])
        return pair.dropna(subset=[dt_feature]).sort_values(dt_feature)

    @track_performance
    def _get_feature_pairs(
        self, cat_columns: list[str], dt_columns: list[str]
    ) -> list[tuple[str, str]]:
        return list(product(cat_columns, dt_columns))

    @track_performance
    def _prepare_top_categories(self, pair: pd.DataFrame, cat_feature: str) -> pd.DataFrame:
        pair = pair.copy()
        top = pair[cat_feature].value_counts().head(self.TOP_N_CATEGORIES).index
        pair[cat_feature] = pair[cat_feature].where(pair[cat_feature].isin(top), "Other")
        return pair

    @track_performance
    def _safely_process_pairs(
        self,
        pairs: list[tuple[str, str]],
        processor: Callable[[str, str], dict[str, Any] | list[dict[str, Any]] | None],
    ) -> pd.DataFrame:
        results: list[dict[str, Any]] = []
        for cat_f, dt_f in pairs:
            try:
                result = processor(cat_f, dt_f)
                if result is None:
                    continue
                results.extend(result if isinstance(result, list) else [result])
            except Exception as exc:
                self.logger.warning("Error processing pair (%s, %s): %s", cat_f, dt_f, exc)
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # 1. Frequency over time
    # ------------------------------------------------------------------
    @track_performance
    def generate_frequency_over_time(
        self,
        categorical_columns: list[str] | None = None,
        datetime_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> pd.DataFrame:
        cat_cols = categorical_columns if categorical_columns is not None else self.get_categorical_columns()
        dt_cols = datetime_columns if datetime_columns is not None else self.get_datetime_columns()
        frequency = frequency if frequency is not None else self.TEMPORAL_FREQUENCY
        pairs = self._get_feature_pairs(cat_cols, dt_cols)
        self.logger.info("Generating category frequency over time for %d pairs.", len(pairs))

        def process_pair(cat_f: str, dt_f: str) -> list[dict[str, Any]] | None:
            pair = self._get_valid_pair(cat_f, dt_f)
            if len(pair) < self.MIN_OBSERVATIONS:
                return None
            pair = self._prepare_top_categories(pair, cat_f)
            pair["period"] = pair[dt_f].dt.to_period(frequency).astype(str)
            grouped = (
                pair.groupby(["period", cat_f], observed=True)
                .size()
                .reset_index(name="frequency")
            )
            if grouped.empty:
                return None
            grouped.insert(0, "categorical_feature", cat_f)
            grouped.insert(1, "datetime_feature", dt_f)
            grouped = grouped.rename(columns={cat_f: "category"})
            return grouped.to_dict(orient="records")

        result = self._safely_process_pairs(pairs, process_pair)
        if result.empty:
            return pd.DataFrame(
                columns=["categorical_feature", "datetime_feature", "period", "category", "frequency"]
            )
        return result[["categorical_feature", "datetime_feature", "period", "category", "frequency"]]

    # ------------------------------------------------------------------
    # 2. Proportion over time
    # ------------------------------------------------------------------
    @track_performance
    def generate_proportion_over_time(
        self,
        categorical_columns: list[str] | None = None,
        datetime_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> pd.DataFrame:
        cat_cols = categorical_columns if categorical_columns is not None else self.get_categorical_columns()
        dt_cols = datetime_columns if datetime_columns is not None else self.get_datetime_columns()
        frequency = frequency if frequency is not None else self.TEMPORAL_FREQUENCY
        pairs = self._get_feature_pairs(cat_cols, dt_cols)
        self.logger.info("Generating category proportions over time.")

        def process_pair(cat_f: str, dt_f: str) -> list[dict[str, Any]] | None:
            pair = self._get_valid_pair(cat_f, dt_f)
            if len(pair) < self.MIN_OBSERVATIONS:
                return None
            pair = self._prepare_top_categories(pair, cat_f)
            pair["period"] = pair[dt_f].dt.to_period(frequency).astype(str)
            counts = (
                pair.groupby(["period", cat_f], observed=True)
                .size()
                .reset_index(name="frequency")
            )
            if counts.empty:
                return None
            counts["period_total"] = counts.groupby("period")["frequency"].transform("sum")
            counts["proportion"] = counts["frequency"] / counts["period_total"]
            counts["percentage"] = counts["proportion"] * 100
            counts.insert(0, "categorical_feature", cat_f)
            counts.insert(1, "datetime_feature", dt_f)
            counts = counts.rename(columns={cat_f: "category"})
            return counts.to_dict(orient="records")

        result = self._safely_process_pairs(pairs, process_pair)
        if result.empty:
            return pd.DataFrame(
                columns=[
                    "categorical_feature", "datetime_feature", "period", "category",
                    "frequency", "period_total", "proportion", "percentage",
                ]
            )
        return result[
            [
                "categorical_feature", "datetime_feature", "period", "category",
                "frequency", "period_total", "proportion", "percentage",
            ]
        ]

    # ------------------------------------------------------------------
    # 3. Chi-square by period
    # ------------------------------------------------------------------
    @track_performance
    def generate_chi_square_by_period(
        self,
        categorical_columns: list[str] | None = None,
        datetime_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> pd.DataFrame:
        cat_cols = categorical_columns if categorical_columns is not None else self.get_categorical_columns()
        dt_cols = datetime_columns if datetime_columns is not None else self.get_datetime_columns()
        frequency = frequency if frequency is not None else self.TEMPORAL_FREQUENCY
        pairs = self._get_feature_pairs(cat_cols, dt_cols)
        self.logger.info("Generating chi-square temporal summaries.")

        def process_pair(cat_f: str, dt_f: str) -> dict[str, Any] | None:
            pair = self._get_valid_pair(cat_f, dt_f)
            if len(pair) < self.MIN_OBSERVATIONS:
                return None
            pair = self._prepare_top_categories(pair, cat_f)
            pair["period"] = pair[dt_f].dt.to_period(frequency).astype(str)
            contingency = pd.crosstab(pair["period"], pair[cat_f])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return None
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            n = int(contingency.to_numpy().sum())
            min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
            cramers_v = (
                np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else np.nan
            )
            return {
                "categorical_feature": cat_f,
                "datetime_feature": dt_f,
                "sample_size": n,
                "number_of_periods": int(contingency.shape[0]),
                "number_of_categories": int(contingency.shape[1]),
                "chi_square": float(chi2),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_val),
                "cramers_v": float(cramers_v) if not pd.isna(cramers_v) else np.nan,
                "min_expected_count": float(expected.min()),
                "expected_count_below_5": int((expected < 5).sum()),
            }

        result = self._safely_process_pairs(pairs, process_pair)
        if not result.empty:
            result = result.sort_values("cramers_v", ascending=False)
        return result

    # ------------------------------------------------------------------
    # 4. Category trend
    # ------------------------------------------------------------------
    @track_performance
    def generate_category_trend(
        self,
        categorical_columns: list[str] | None = None,
        datetime_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> pd.DataFrame:
        cat_cols = categorical_columns if categorical_columns is not None else self.get_categorical_columns()
        dt_cols = datetime_columns if datetime_columns is not None else self.get_datetime_columns()
        frequency = frequency if frequency is not None else self.TEMPORAL_FREQUENCY
        pairs = self._get_feature_pairs(cat_cols, dt_cols)
        self.logger.info("Generating category trend summaries.")

        results: list[dict[str, Any]] = []
        for cat_f, dt_f in pairs:
            try:
                pair = self._get_valid_pair(cat_f, dt_f)
                if len(pair) < self.MIN_OBSERVATIONS:
                    continue
                pair = self._prepare_top_categories(pair, cat_f)
                pair["period"] = pair[dt_f].dt.to_period(frequency)
                grouped = (
                    pair.groupby(["period", cat_f], observed=True)
                    .size()
                    .reset_index(name="frequency")
                )
                if grouped.empty or grouped["period"].nunique() < self.MIN_PERIODS:
                    continue
                grouped["period_total"] = grouped.groupby("period")["frequency"].transform("sum")
                grouped["proportion"] = grouped["frequency"] / grouped["period_total"]
                pivot = grouped.pivot_table(
                    index=cat_f, columns="period", values="proportion",
                    aggfunc="mean", fill_value=0,
                )
                if pivot.shape[1] < self.MIN_PERIODS:
                    continue
                first_period, last_period = pivot.columns[0], pivot.columns[-1]
                for category in pivot.index:
                    vals = pivot.loc[category]
                    first_prop = float(vals.iloc[0])
                    last_prop = float(vals.iloc[-1])
                    change = last_prop - first_prop
                    results.append({
                        "categorical_feature": cat_f,
                        "datetime_feature": dt_f,
                        "category": str(category),
                        "first_period": str(first_period),
                        "last_period": str(last_period),
                        "first_period_proportion": first_prop,
                        "last_period_proportion": last_prop,
                        "change_in_proportion": change,
                        "change_percentage_points": change * 100,
                        "average_proportion": float(vals.mean()),
                        "minimum_proportion": float(vals.min()),
                        "maximum_proportion": float(vals.max()),
                    })
            except Exception as exc:
                self.logger.warning(
                    "Error generating category trend for (%s, %s): %s", cat_f, dt_f, exc
                )

        result = pd.DataFrame(results)
        if not result.empty:
            result = result.sort_values(
                "change_in_proportion", key=lambda v: v.abs(), ascending=False
            )
        return result

    # ------------------------------------------------------------------
    # Excel formatting & writer
    # ------------------------------------------------------------------
    @staticmethod
    @track_performance
    def format_excel_workbook(writer: pd.ExcelWriter) -> None:
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
                cell.alignment = cell.alignment.copy(horizontal="center")
            for col_cells in ws.columns:
                if not col_cells:
                    continue
                max_len = 0
                letter = col_cells[0].column_letter
                for cell in col_cells[:100]:
                    try:
                        max_len = max(max_len, len(str(cell.value)))
                    except Exception:
                        continue
                ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 30)

    @track_performance
    def save_workbook(self, results: dict[str, pd.DataFrame]) -> Path:
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = self.REPORT_DIR / self.WORKBOOK_NAME
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for name, df in results.items():
                    df.to_excel(writer, sheet_name=name, index=False)
                self.format_excel_workbook(writer)
            self.logger.info("Workbook successfully created: %s", path)
            return path
        except Exception as exc:
            self.logger.error("Failed to save CAT-DATETIME workbook: %s", exc)
            raise CustomException(f"Failed to save CAT-DATETIME workbook: {exc}") from exc

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    @track_performance
    def run_eda(self, df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
        if df is not None:
            self.df = df
        self._validate_dataframe()
        cat_cols = self.get_categorical_columns()
        dt_cols = self.get_datetime_columns()
        if not cat_cols:
            raise CustomException("No categorical columns available for CAT-DATETIME EDA.")
        if not dt_cols:
            raise CustomException("No datetime columns available for CAT-DATETIME EDA.")
        self.logger.info("Starting CAT-DATETIME exploratory data analysis.")
        self.logger.info("Categorical columns: %s", cat_cols)
        self.logger.info("Datetime columns: %s", dt_cols)

        results = {
            "frequency_over_time": self.generate_frequency_over_time(
                cat_cols, dt_cols, frequency=self.TEMPORAL_FREQUENCY
            ),
            "proportion_over_time": self.generate_proportion_over_time(
                cat_cols, dt_cols, frequency=self.TEMPORAL_FREQUENCY
            ),
            "chi_square_by_period": self.generate_chi_square_by_period(
                cat_cols, dt_cols, frequency=self.TEMPORAL_FREQUENCY
            ),
            "category_trend": self.generate_category_trend(
                cat_cols, dt_cols, frequency=self.TEMPORAL_FREQUENCY
            ),
        }
        path = self.save_workbook(results)
        self.logger.info(
            "CAT-DATETIME EDA pipeline completed successfully. Workbook: %s", path
        )
        return results

    @track_performance
    def run_from_file(self, file_path: Path) -> dict[str, pd.DataFrame]:
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
        except Exception as exc:
            raise CustomException(f"Failed to execute CAT-DATETIME EDA: {exc}") from exc


@track_performance
def cat_datetime_stats(file_path: Path | None = None) -> dict[str, pd.DataFrame]:
    if file_path is None:
        file_path = BASE_DIR / "src" / "Data" / "cleaned_ecommerce_dataset.csv"
    return CategoricalDatetimeEDA().run_from_file(Path(file_path))


if __name__ == "__main__":
    cat_datetime_stats()