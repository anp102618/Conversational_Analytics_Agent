"""
Categorical-Categorical Bivariate Exploratory Data Analysis Engine.

Computes statistical relationships between categorical variables (contingency tables,
Chi-square tests, Cramér's V, Fisher's exact tests) and exports results to Excel.
"""

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from src.config import BASE_DIR
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class CategoricalCategoricalEDA:
    """Perform bivariate EDA between categorical variables."""

    WORKBOOK_NAME = "cat_cat_eda.xlsx"
    MIN_CATEGORIES = 2
    MIN_SAMPLE_SIZE = 1

    REPORT_DIR = BASE_DIR / "src" / "EDA_Summary" / "Bivariate_Analysis" / "Reports" / "excel_reports"
    LOG_DIR = BASE_DIR / "src" / "EDA_Summary" / "Bivariate_Analysis" / "logs"
    CLASSIFIED_COLUMNS_PATH = BASE_DIR / "src" / "Data" / "classified_columns.json"

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.df = df
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = get_log("CategoricalCategoricalEDA", log_dir=self.LOG_DIR)

    # ============================================================
    # Column Resolution & Pair Processing Utilities
    # ============================================================
    @track_performance
    def load_classified_columns(self) -> dict[str, Any]:
        """Load classified column metadata from JSON."""
        if not self.CLASSIFIED_COLUMNS_PATH.exists():
            self.logger.warning("classified_columns.json not found at %s", self.CLASSIFIED_COLUMNS_PATH)
            return {}
        try:
            with self.CLASSIFIED_COLUMNS_PATH.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            raise CustomException(f"Failed to read classified columns: {exc}") from exc

    @track_performance
    def get_categorical_columns(self) -> list[str]:
        """Resolve categorical columns using JSON classification or dtype fallback."""
        if self.df is None:
            raise CustomException("DataFrame is required before resolving categorical columns.")

        classified = self.load_classified_columns()
        configured = classified.get("bivariate_candidates", {}).get("categorical", [])
        categorical_columns = [col for col in configured if col in self.df.columns]

        if categorical_columns:
            self.logger.info("Resolved %d categorical columns from classification.", len(categorical_columns))
            return categorical_columns

        fallback = self.df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        self.logger.info("Using pandas dtype fallback: %d columns found.", len(fallback))
        return fallback

    @track_performance
    def _validate_dataframe(self) -> None:
        """Validate DataFrame usability."""
        if self.df is None or self.df.empty:
            raise CustomException("DataFrame cannot be None or empty for CAT-CAT EDA.")
        if self.df.shape[1] < 2:
            raise CustomException("CAT-CAT EDA requires at least two columns.")

    @track_performance
    def _get_contingency_table(self, feature_1: str, feature_2: str) -> pd.DataFrame:
        """Build frequency contingency table after dropping null values."""
        if self.df is None:
            raise CustomException("DataFrame is not initialized.")
        pair = self.df[[feature_1, feature_2]].dropna()
        return pd.crosstab(pair[feature_1], pair[feature_2], dropna=False) if not pair.empty else pd.DataFrame()

    @track_performance
    def _safely_process_pairs(
        self,
        pairs: list[tuple[str, str]],
        processor: Callable[[str, str], dict[str, Any] | list[dict[str, Any]] | None],
    ) -> pd.DataFrame:
        """Iterate over pairs and aggregate non-null results with exception handling."""
        results: list[dict[str, Any]] = []
        for f1, f2 in pairs:
            try:
                res = processor(f1, f2)
                if res is not None:
                    results.extend(res) if isinstance(res, list) else results.append(res)
            except Exception as exc:
                self.logger.warning("Error processing pair (%s, %s): %s", f1, f2, exc)
        return pd.DataFrame(results)

    @track_performance
    @staticmethod
    def _validate_contingency_table(table: pd.DataFrame) -> bool:
        """Verify contingency table dimensions and total counts."""
        return not table.empty and table.shape[0] >= 2 and table.shape[1] >= 2 and table.to_numpy().sum() > 0

    @track_performance
    @staticmethod
    def _calculate_cramers_v(chi_square: float, sample_size: int, table_shape: tuple[int, int]) -> float | None:
        """Compute Cramér's V metric."""
        min_dim = min(table_shape[0] - 1, table_shape[1] - 1)
        if sample_size <= 0 or min_dim <= 0:
            return None
        return float(np.sqrt((chi_square / sample_size) / min_dim))

    # ============================================================
    # Analysis Generators
    # ============================================================
    @track_performance
    def generate_contingency_summary(self, categorical_columns: list[str] | None = None) -> pd.DataFrame:
        """Generate cell counts and overall, row, and column percentages for pairs."""
        cols = categorical_columns or self.get_categorical_columns()
        pairs = list(combinations(cols, 2))
        self.logger.info("Computing contingency summaries for %d pairs.", len(pairs))

        def process_pair(f1: str, f2: str) -> list[dict[str, Any]] | None:
            table = self._get_contingency_table(f1, f2)
            total = int(table.to_numpy().sum()) if not table.empty else 0
            if total < self.MIN_SAMPLE_SIZE:
                return None

            row_totals, col_totals = table.sum(axis=1), table.sum(axis=0)
            results = []
            for c1 in table.index:
                for c2 in table.columns:
                    count = int(table.loc[c1, c2])
                    rt, ct = int(row_totals.loc[c1]), int(col_totals.loc[c2])
                    results.append({
                        "feature_1": f1, "feature_2": f2,
                        "category_1": str(c1), "category_2": str(c2),
                        "count": count,
                        "overall_percentage": (count / total * 100) if total > 0 else 0.0,
                        "row_total": rt,
                        "row_percentage": (count / rt * 100) if rt > 0 else 0.0,
                        "column_total": ct,
                        "column_percentage": (count / ct * 100) if ct > 0 else 0.0,
                    })
            return results

        return self._safely_process_pairs(pairs, process_pair)

    @track_performance
    def generate_chi_square(self, categorical_columns: list[str] | None = None) -> pd.DataFrame:
        """Compute Chi-square tests of independence and diagnostics."""
        cols = categorical_columns or self.get_categorical_columns()
        pairs = list(combinations(cols, 2))
        self.logger.info("Computing Chi-square tests for %d pairs.", len(pairs))

        def process_pair(f1: str, f2: str) -> dict[str, Any] | None:
            table = self._get_contingency_table(f1, f2)
            if not self._validate_contingency_table(table):
                return None
            chi2, p_val, dof, expected = chi2_contingency(table)
            return {
                "feature_1": f1, "feature_2": f2,
                "sample_size": int(table.to_numpy().sum()),
                "rows": int(table.shape[0]), "columns": int(table.shape[1]),
                "chi_square_statistic": float(chi2),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_val),
                "min_expected_count": float(expected.min()),
                "expected_count_below_5": int((expected < 5).sum()),
                "expected_count_below_1": int((expected < 1).sum()),
            }

        return self._safely_process_pairs(pairs, process_pair)

    @track_performance
    def generate_cramers_v(self, categorical_columns: list[str] | None = None) -> pd.DataFrame:
        """Calculate Cramér's V association strength for pairs."""
        cols = categorical_columns or self.get_categorical_columns()
        pairs = list(combinations(cols, 2))
        self.logger.info("Computing Cramér's V for %d pairs.", len(pairs))

        def process_pair(f1: str, f2: str) -> dict[str, Any] | None:
            table = self._get_contingency_table(f1, f2)
            if not self._validate_contingency_table(table):
                return None
            chi2, p_val, dof, _ = chi2_contingency(table)
            n = int(table.to_numpy().sum())
            c_v = self._calculate_cramers_v(chi2, n, table.shape)
            if c_v is None:
                return None
            return {
                "feature_1": f1, "feature_2": f2, "sample_size": n,
                "chi_square_statistic": float(chi2), "degrees_of_freedom": int(dof),
                "p_value": float(p_val), "cramers_v": c_v,
            }

        return self._safely_process_pairs(pairs, process_pair)

    @track_performance
    def generate_fisher_exact(self, categorical_columns: list[str] | None = None) -> pd.DataFrame:
        """Perform Fisher's exact test for 2x2 categorical tables."""
        cols = categorical_columns or self.get_categorical_columns()
        pairs = list(combinations(cols, 2))
        self.logger.info("Computing Fisher's exact tests.")

        def process_pair(f1: str, f2: str) -> dict[str, Any] | None:
            table = self._get_contingency_table(f1, f2)
            if table.shape != (2, 2):
                return None
            odds_ratio, p_val = fisher_exact(table.to_numpy())
            return {
                "feature_1": f1, "feature_2": f2,
                "category_1_level_1": str(table.index[0]),
                "category_1_level_2": str(table.index[1]),
                "category_2_level_1": str(table.columns[0]),
                "category_2_level_2": str(table.columns[1]),
                "n_11": int(table.iloc[0, 0]), "n_12": int(table.iloc[0, 1]),
                "n_21": int(table.iloc[1, 0]), "n_22": int(table.iloc[1, 1]),
                "odds_ratio": float(odds_ratio), "p_value": float(p_val),
            }

        return self._safely_process_pairs(pairs, process_pair)

    # ============================================================
    # Execution & Output
    # ============================================================

    @track_performance
    def save_workbook(self, results: dict[str, pd.DataFrame]) -> Path:
        """Save results mapping to multi-sheet Excel workbook."""
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = self.REPORT_DIR / self.WORKBOOK_NAME
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for sheet, df in results.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)
            self.logger.info("CAT-CAT workbook created: %s", path)
            return path
        except Exception as exc:
            self.logger.error("Failed to save CAT-CAT workbook: %s", exc)
            raise CustomException(f"Failed to save CAT-CAT EDA workbook: {exc}") from exc

    @track_performance
    def run_eda(self, df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
        """Execute full CAT-CAT EDA pipeline and export results."""
        if df is not None:
            self.df = df
        self._validate_dataframe()

        cat_cols = self.get_categorical_columns()
        if len(cat_cols) < 2:
            raise CustomException("At least two categorical columns are required for CAT-CAT EDA.")

        self.logger.info("Starting CAT-CAT EDA with columns: %s", cat_cols)
        results = {
            "contingency_summary": self.generate_contingency_summary(cat_cols),
            "chi_square": self.generate_chi_square(cat_cols),
            "cramers_v": self.generate_cramers_v(cat_cols),
            "fisher_exact": self.generate_fisher_exact(cat_cols),
        }
        self.save_workbook(results)
        return results

    @track_performance
    def run_from_file(self, file_path: Path) -> dict[str, pd.DataFrame]:
        """Load CSV dataset and run complete EDA pipeline."""
        path = Path(file_path)
        self.logger.info("Loading dataset from: %s", path)
        if not path.exists():
            raise CustomException(f"Dataset file does not exist: {path}")

        try:
            df = pd.read_csv(path)
            self.logger.info("Dataset loaded with shape: %s", df.shape)
            return self.run_eda(df)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception) as exc:
            raise CustomException(f"Failed to execute CAT-CAT EDA from file: {exc}") from exc


@track_performance
def cat_cat_stats(file_path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Module entry point wrapper function."""
    path = file_path or (BASE_DIR / "src" / "Data" / "cleaned_ecommerce_dataset.csv")
    return CategoricalCategoricalEDA().run_from_file(Path(path))


if __name__ == "__main__":
    cat_cat_stats()