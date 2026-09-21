import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

from src.config import BASE_DIR
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class NumericNumericEDA:
    """Class-based engine for numeric-numeric bivariate EDA."""

    WORKBOOK_NAME = "num_num_eda.xlsx"
    LOW_SAMPLE_THRESHOLD = 3
    MI_SAMPLE_THRESHOLD = 10

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.df = df
        base_path = BASE_DIR / "src"
        self.reports_dir = base_path / "EDA_Summary" / "Bivariate_Analysis" / "Reports" / "excel_reports"
        self.logs_dir = base_path / "EDA_Summary" / "Bivariate_Analysis" / "logs"
        self.columns_json_path = base_path / "Data" / "classified_columns.json"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_log("BivariateNUMNUMEDA", log_dir=self.logs_dir)

    @track_performance
    def load_classified_columns(self) -> dict[str, Any]:
        """Load classified column metadata from JSON."""
        try:
            with self.columns_json_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            self.logger.error("Failed to load classified columns from %s: %s", self.columns_json_path, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def get_numeric_columns(self) -> list[str]:
        """Resolve numeric columns from metadata or fall back to DataFrame dtypes."""
        try:
            columns = self.load_classified_columns()["univariate_candidates"]["numeric"]
            if not isinstance(columns, list):
                raise ValueError("'univariate_candidates.numeric' must be a list.")
            
            valid_cols = [col for col in columns if self.df is not None and col in self.df.columns]
            if valid_cols:
                return valid_cols
            raise ValueError("No classified numeric columns exist in the DataFrame.")
        except Exception as exc:
            self.logger.warning("Could not load numeric columns from JSON (%s). Falling back to dtypes.", exc)
            if self.df is None:
                raise CustomException("DataFrame required for fallback detection.", self.logger) from exc
            
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            return [col for col in num_cols if not pd.api.types.is_bool_dtype(self.df[col])]

    @staticmethod
    def _get_valid_pair(df: pd.DataFrame, feature_1: str, feature_2: str) -> pd.DataFrame:
        """Return pairwise valid numeric observations with infinite values coersed to NaN."""
        pair = df[[feature_1, feature_2]].apply(pd.to_numeric, errors="coerce")
        return pair.replace([np.inf, -np.inf], np.nan).dropna()

    @track_performance
    def _safely_process_pairs(
        self,
        numeric_columns: list[str],
        process_fn: Callable[[str, str], dict[str, Any] | None],
        metric_name: str,
    ) -> pd.DataFrame:
        """Safely process every unique numeric feature pair."""
        results: list[dict[str, Any]] = []
        for i, feature_1 in enumerate(numeric_columns):
            for feature_2 in numeric_columns[i + 1:]:
                try:
                    if result := process_fn(feature_1, feature_2):
                        results.append(result)
                    self.logger.info("Computed %s for pair: (%s, %s)", metric_name, feature_1, feature_2)
                except Exception as exc:
                    self.logger.warning("Error computing %s for (%s, %s): %s", metric_name, feature_1, feature_2, exc)
        return pd.DataFrame(results)

    @track_performance
    def generate_pearson_correlation(self, numeric_columns: list[str]) -> pd.DataFrame:
        """Generate Pearson correlation statistics."""
        self.logger.info("Computing Pearson correlations for %d numeric features...", len(numeric_columns))

        def _calculate(f1: str, f2: str) -> dict[str, Any] | None:
            pair = self._get_valid_pair(self.df, f1, f2)
            if len(pair) < self.LOW_SAMPLE_THRESHOLD or pair[f1].nunique() < 2 or pair[f2].nunique() < 2:
                return None
            corr, p_val = pearsonr(pair[f1], pair[f2])
            return {
                "feature_1": f1, "feature_2": f2, "sample_size": len(pair),
                "pearson_correlation": corr, "p_value": p_val, "absolute_correlation": abs(corr),
            }

        return self._safely_process_pairs(numeric_columns, _calculate, "Pearson correlation")

    @track_performance
    def generate_spearman_correlation(self, numeric_columns: list[str]) -> pd.DataFrame:
        """Generate Spearman rank correlation statistics."""
        self.logger.info("Computing Spearman correlations for %d numeric features...", len(numeric_columns))

        def _calculate(f1: str, f2: str) -> dict[str, Any] | None:
            pair = self._get_valid_pair(self.df, f1, f2)
            if len(pair) < self.LOW_SAMPLE_THRESHOLD or pair[f1].nunique() < 2 or pair[f2].nunique() < 2:
                return None
            corr, p_val = spearmanr(pair[f1], pair[f2])
            return {
                "feature_1": f1, "feature_2": f2, "sample_size": len(pair),
                "spearman_correlation": corr, "p_value": p_val, "absolute_correlation": abs(corr),
            }

        return self._safely_process_pairs(numeric_columns, _calculate, "Spearman correlation")

    @track_performance
    def generate_kendall_correlation(self, numeric_columns: list[str]) -> pd.DataFrame:
        """Generate Kendall's Tau correlation statistics."""
        self.logger.info("Computing Kendall correlations for %d numeric features...", len(numeric_columns))

        def _calculate(f1: str, f2: str) -> dict[str, Any] | None:
            pair = self._get_valid_pair(self.df, f1, f2)
            if len(pair) < self.LOW_SAMPLE_THRESHOLD or pair[f1].nunique() < 2 or pair[f2].nunique() < 2:
                return None
            corr, p_val = kendalltau(pair[f1], pair[f2])
            return {
                "feature_1": f1, "feature_2": f2, "sample_size": len(pair),
                "kendall_tau": corr, "p_value": p_val, "absolute_tau": abs(corr),
            }

        return self._safely_process_pairs(numeric_columns, _calculate, "Kendall correlation")

    @track_performance
    def generate_mutual_information(self, numeric_columns: list[str]) -> pd.DataFrame:
        """Generate symmetric mutual information between numeric pairs."""
        self.logger.info("Computing Mutual Information for %d numeric features...", len(numeric_columns))

        def _calculate(f1: str, f2: str) -> dict[str, Any] | None:
            pair = self._get_valid_pair(self.df, f1, f2)
            if len(pair) < self.MI_SAMPLE_THRESHOLD or pair[f1].nunique() < 2 or pair[f2].nunique() < 2:
                return None

            x = pair[f1].to_numpy().reshape(-1, 1)
            y = pair[f2].to_numpy()

            mi_xy = mutual_info_regression(x, y, random_state=42)[0]
            mi_yx = mutual_info_regression(y.reshape(-1, 1), pair[f1].to_numpy(), random_state=42)[0]

            return {
                "feature_1": f1, "feature_2": f2, "sample_size": len(pair),
                "mutual_information": (mi_xy + mi_yx) / 2,
                "mi_feature_1_to_feature_2": mi_xy, "mi_feature_2_to_feature_1": mi_yx,
            }

        return self._safely_process_pairs(numeric_columns, _calculate, "Mutual Information")

    @track_performance
    def save_workbook(self, results: dict[str, pd.DataFrame]) -> None:
        """Save all NUM-NUM EDA results into one Excel workbook."""
        try:
            workbook_path = self.reports_dir / self.WORKBOOK_NAME
            with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
                for sheet_name, dataframe in results.items():
                    dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

            self.logger.info("Workbook successfully created at: %s (%d sheets)", workbook_path, len(results))
        except Exception as exc:
            self.logger.error("Failed to save NUM-NUM workbook: %s", exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def run_eda(self, df: pd.DataFrame | None = None) -> None:
        """Execute the complete NUM-NUM EDA pipeline."""
        if df is not None:
            self.df = df

        if self.df is None or self.df.empty:
            raise CustomException("DataFrame is empty. Please provide a valid pandas DataFrame.", self.logger)

        self.logger.info("Starting NUMERIC-NUMERIC exploratory data analysis...")
        numeric_columns = self.get_numeric_columns()

        if len(numeric_columns) < 2:
            raise CustomException("At least two numeric columns are required for NUM-NUM analysis.", self.logger)

        self.logger.info("Resolved %d numeric columns for NUM-NUM EDA.", len(numeric_columns))

        metrics_map = {
            "pearson": self.generate_pearson_correlation,
            "spearman": self.generate_spearman_correlation,
            "kendall": self.generate_kendall_correlation,
            "mutual_information": self.generate_mutual_information,
        }

        results = {sheet: func(numeric_columns) for sheet, func in metrics_map.items()}
        self.save_workbook(results)
        self.logger.info("NUMERIC-NUMERIC EDA pipeline completed successfully.")

    @track_performance
    def run_from_file(self, file_path: Path | str) -> None:
        """Load a CSV file and execute NUM-NUM EDA."""
        file_path = Path(file_path)
        self.logger.info("Attempting to load dataset from: %s", file_path)

        if not file_path.exists():
            raise CustomException(FileNotFoundError(f"The file {file_path} does not exist."), self.logger)

        try:
            self.df = pd.read_csv(file_path)
            self.logger.info("Dataset successfully loaded with shape: %s", self.df.shape)
            self.run_eda()
        except pd.errors.EmptyDataError as exc:
            self.logger.error("Dataset is empty: %s", file_path)
            raise CustomException(exc, self.logger) from exc
        except Exception as exc:
            self.logger.error("Error processing CSV execution: %s", exc)
            raise CustomException(exc, self.logger) from exc


@track_performance
def num_num_stats(file_path: Path | str | None = None) -> None:
    """Wrapper for executing numeric-numeric bivariate EDA."""
    if file_path is None:
        file_path = BASE_DIR / "src" / "Data" / "cleaned_ecommerce_dataset.csv"

    eda_engine = NumericNumericEDA()
    eda_engine.run_from_file(file_path=file_path)


if __name__ == "__main__":
    num_num_stats()