from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException


class CategoricalUnivariateEDA:
    """Class-based engine for categorical univariate exploratory data analysis (EDA)."""

    RARE_CATEGORY_THRESHOLD = 1.0
    TOP_N_CATEGORIES = 10
    WORKBOOK_NAME = "categorical_uni_eda.xlsx"

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        rare_category_threshold: float = RARE_CATEGORY_THRESHOLD,
        top_n: int = TOP_N_CATEGORIES,
    ) -> None:
        self.df = df
        self.rare_category_threshold = rare_category_threshold
        self.top_n = top_n

        # Directory paths setup
        self.reports_dir = BASE_DIR / "src" / "EDA_Summary" / "Univariate_Analysis" / "Reports" / "excel_reports"
        self.logs_dir = BASE_DIR / "src" / "EDA_Summary" / "Univariate_Analysis" / "logs"
        self.columns_json_path = BASE_DIR / "src" / "Data" / "classified_columns.json"

        # Ensure directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_log("UnivariateCATEDA", log_dir=self.logs_dir)
        self._processed_series_cache: Dict[str, pd.Series] = {}

    # ============================================================
    # Helpers & Column Resolution
    # ============================================================
    @track_performance
    def load_classified_columns(self) -> dict:
        try:
            with open(self.columns_json_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            self.logger.error(f"Failed to load classified columns from {self.columns_json_path}: {e}")
            raise CustomException(e, self.logger)

    @track_performance
    def get_categorical_columns(self) -> List[str]:
        try:
            columns_dict = self.load_classified_columns()
            candidates = []
            if "univariate_candidates" in columns_dict:
                candidates = columns_dict["univariate_candidates"].get("categorical", [])
            elif "categorical" in columns_dict:
                candidates = columns_dict.get("categorical", [])
            elif "bivariate_candidates" in columns_dict:
                candidates = columns_dict["bivariate_candidates"].get("categorical", [])

            if candidates and self.df is not None:
                valid_candidates = [c for c in candidates if c in self.df.columns]
                if valid_candidates:
                    self.logger.info(f"Loaded {len(valid_candidates)} categorical columns from JSON config")
                    return valid_candidates
        except Exception as e:
            self.logger.warning(f"Could not load columns from JSON ({e}). Falling back to DataFrame dtype detection.")

        if self.df is not None:
            cat_cols = self.df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            self.logger.info(f"Detected {len(cat_cols)} categorical columns via dtype selection")
            return cat_cols

        raise CustomException("DataFrame must be provided when fallback detection is triggered.", self.logger)

    def _get_cleaned_series(self, column: str) -> pd.Series:
        """Retrieves or caches string-cleaned series with stripped whitespace converted to NaN."""
        if column not in self._processed_series_cache:
            series = self.df[column].astype("string")
            self._processed_series_cache[column] = series.mask(series.str.strip().eq(""))
        return self._processed_series_cache[column]

    def _valid_series(self, column: str) -> pd.Series:
        return self._get_cleaned_series(column).dropna()

    # ============================================================
    # Calculations & Computations
    # ============================================================
    @track_performance
    def generate_basic_statistics(self, cat_cols: List[str]) -> pd.DataFrame:
        self.logger.info(f"Computing basic statistics for {len(cat_cols)} categorical features...")
        results = []

        for col in cat_cols:
            try:
                series = self._get_cleaned_series(col)
                total_rows = len(series)
                missing_count = int(series.isna().sum())
                valid_count = total_rows - missing_count
                unique_count = int(series.nunique(dropna=True))

                results.append({
                    "feature": col,
                    "dtype": str(self.df[col].dtype),
                    "total_rows": total_rows,
                    "valid_count": valid_count,
                    "missing_count": missing_count,
                    "missing_percentage": (missing_count / total_rows * 100) if total_rows else 0.0,
                    "unique_count": unique_count,
                    "unique_percentage": (unique_count / valid_count * 100) if valid_count else 0.0,
                })
            except Exception as e:
                self.logger.warning(f"Error computing basic statistics for {col}: {e}")

        return pd.DataFrame(results)

    @track_performance
    def generate_distribution_summary(self, cat_cols: List[str]) -> pd.DataFrame:
        """Unified calculation for frequency, percentage, and cumulative percentage distribution."""
        self.logger.info("Computing cumulative and percentage distributions...")
        results = []

        for col in cat_cols:
            try:
                series = self._valid_series(col)
                if series.empty:
                    continue
                
                counts = series.value_counts(dropna=True)
                total = len(series)
                cumulative = 0.0

                for category, count in counts.items():
                    pct = (count / total) * 100
                    cumulative += pct
                    results.append({
                        "feature": col,
                        "category": str(category),
                        "count": int(count),
                        "percentage": pct,
                        "cumulative_percentage": cumulative,
                    })
            except Exception as e:
                self.logger.warning(f"Error computing distribution for {col}: {e}")

        return pd.DataFrame(results)

    @track_performance
    def generate_mode_statistics(self, cat_cols: List[str]) -> pd.DataFrame:
        self.logger.info("Computing mode statistics...")
        results = []

        for col in cat_cols:
            try:
                series = self._valid_series(col)
                if series.empty:
                    results.append({
                        "feature": col,
                        "mode": None,
                        "mode_count": 0,
                        "mode_percentage": 0.0,
                    })
                    continue
                
                counts = series.value_counts()
                max_count = counts.max()
                modes = counts[counts == max_count]
                total = len(series)

                for mode, count in modes.items():
                    results.append({
                        "feature": col,
                        "mode": str(mode),
                        "mode_count": int(count),
                        "mode_percentage": (count / total * 100),
                    })
            except Exception as e:
                self.logger.warning(f"Error computing mode statistics for {col}: {e}")

        return pd.DataFrame(results)

    @track_performance
    def generate_cardinality_statistics(self, cat_cols: List[str]) -> pd.DataFrame:
        self.logger.info("Computing cardinality statistics...")
        results = []

        for col in cat_cols:
            try:
                series = self._valid_series(col)
                valid_count = len(series)
                unique_count = int(series.nunique())
                cardinality_percentage = (unique_count / valid_count * 100) if valid_count else 0.0

                results.append({
                    "feature": col,
                    "valid_count": valid_count,
                    "unique_count": unique_count,
                    "cardinality_percentage": cardinality_percentage,
                    "is_binary": unique_count == 2,
                    "is_single_value": unique_count <= 1,
                    "is_high_cardinality": (cardinality_percentage >= 50.0),
                    "is_id_like": (valid_count > 0 and unique_count / valid_count >= 0.95),
                })
            except Exception as e:
                self.logger.warning(f"Error computing cardinality statistics for {col}: {e}")

        return pd.DataFrame(results)

    @track_performance
    def generate_rare_categories(self, cat_cols: List[str]) -> pd.DataFrame:
        self.logger.info("Computing rare categories...")
        results = []

        for col in cat_cols:
            try:
                series = self._valid_series(col)
                if series.empty:
                    continue
                
                counts = series.value_counts()
                total = len(series)
                for category, count in counts.items():
                    percentage = (count / total) * 100
                    if percentage < self.rare_category_threshold:
                        results.append({
                            "feature": col,
                            "category": str(category),
                            "count": int(count),
                            "percentage": percentage,
                            "threshold_percentage": self.rare_category_threshold,
                        })
            except Exception as e:
                self.logger.warning(f"Error computing rare categories for {col}: {e}")

        return pd.DataFrame(results)

    @track_performance
    def generate_top_categories(self, cat_cols: List[str]) -> pd.DataFrame:
        self.logger.info("Computing top categories...")
        results = []

        for col in cat_cols:
            try:
                series = self._valid_series(col)
                if series.empty:
                    continue
                
                counts = series.value_counts().head(self.top_n)
                total = len(series)
                for rank, (category, count) in enumerate(counts.items(), start=1):
                    results.append({
                        "feature": col,
                        "rank": rank,
                        "category": str(category),
                        "count": int(count),
                        "percentage": (count / total) * 100,
                    })
            except Exception as e:
                self.logger.warning(f"Error computing top categories for {col}: {e}")

        return pd.DataFrame(results)

    @track_performance
    def generate_entropy_statistics(self, cat_cols: List[str]) -> pd.DataFrame:
        self.logger.info("Computing entropy statistics...")
        results = []

        for col in cat_cols:
            try:
                series = self._valid_series(col)
                if series.empty:
                    results.append({
                        "feature": col,
                        "entropy": np.nan,
                        "normalized_entropy": np.nan,
                        "unique_count": 0,
                    })
                    continue

                probabilities = series.value_counts(normalize=True).to_numpy()
                entropy = float(-np.sum(probabilities * np.log2(probabilities)))
                unique_count = len(probabilities)
                normalized_entropy = entropy / np.log2(unique_count) if unique_count > 1 else 0.0

                results.append({
                    "feature": col,
                    "entropy": entropy,
                    "normalized_entropy": normalized_entropy,
                    "unique_count": unique_count,
                })
            except Exception as e:
                self.logger.warning(f"Error computing entropy statistics for {col}: {e}")

        return pd.DataFrame(results)

    # ============================================================
    # Saving & Pipeline Execution
    # ============================================================
    @track_performance
    def save_workbook(self, results: dict[str, pd.DataFrame]) -> None:
        try:
            wb_path = self.reports_dir / self.WORKBOOK_NAME
            with pd.ExcelWriter(wb_path, engine="openpyxl") as writer:
                for sheet, df in results.items():
                    # Truncate sheet name to 31 chars to prevent openpyxl error
                    clean_sheet = sheet[:31]
                    df.to_excel(writer, sheet_name=clean_sheet, index=False)
            self.logger.info(f"Workbook successfully created at: {wb_path} ({len(results)} sheets)")
        except Exception as e:
            self.logger.error(f"Failed to save Excel workbook: {e}")
            raise CustomException(e, self.logger)

    @track_performance
    def run_eda(self, df: pd.DataFrame | None = None) -> None:
        if df is not None:
            self.df = df

        if self.df is None or self.df.empty:
            raise CustomException("DataFrame is empty. Please pass a valid pandas DataFrame.", self.logger)

        self.logger.info("Starting CATEGORICAL-UNI exploratory data analysis...")
        
        # Clear series cache prior to new run
        self._processed_series_cache.clear()
        
        cat_cols = self.get_categorical_columns()

        # Excel sheet names kept <= 31 characters
        results = {
            "basic_statistics": self.generate_basic_statistics(cat_cols),
            "distribution_summary": self.generate_distribution_summary(cat_cols),
            "mode_statistics": self.generate_mode_statistics(cat_cols),
            "cardinality_stats": self.generate_cardinality_statistics(cat_cols),
            "rare_categories": self.generate_rare_categories(cat_cols),
            "top_categories": self.generate_top_categories(cat_cols),
            "entropy_statistics": self.generate_entropy_statistics(cat_cols),
        }
        
        self.save_workbook(results)
        self.logger.info("CATEGORICAL-UNI EDA pipeline completed successfully.")

    @track_performance
    def run_from_file(self, file_path: Path | str) -> None:
        file_path = Path(file_path)
        self.logger.info(f"Attempting to load dataset from: {file_path}")
        if not file_path.exists():
            self.logger.error(f"The file {file_path} does not exist.")
            raise CustomException(FileNotFoundError(f"The file {file_path} does not exist."), self.logger)

        try:
            self.df = pd.read_csv(file_path)
            self.logger.info(f"Dataset successfully loaded with shape: {self.df.shape}")
            self.run_eda()
        except Exception as e:
            self.logger.error(f"Error processing CSV execution: {e}")
            raise CustomException(e, self.logger)


# ============================================================
# Entry Point & Function Wrapper
# ============================================================
@track_performance
def uni_cat_stats(file_path: Path | str | None = None) -> None:
    """Wrapper function to instantiate and run categorical univariate statistics."""
    if file_path is None:
        file_path = BASE_DIR / "src" / "Data" / "cleaned_ecommerce_dataset.csv"

    eda_engine = CategoricalUnivariateEDA()
    eda_engine.run_from_file(file_path=file_path)


if __name__ == "__main__":
    uni_cat_stats()