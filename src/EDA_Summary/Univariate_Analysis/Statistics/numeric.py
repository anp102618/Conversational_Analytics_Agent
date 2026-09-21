import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import anderson, kurtosis, normaltest, skew

from src.config import BASE_DIR
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class NumericUnivariateEDA:
    """Class-based engine for numeric univariate exploratory data analysis (EDA)."""

    HISTOGRAM_BINS = 20
    LOW_CARDINALITY_THRESHOLD = 10
    WORKBOOK_NAME = "numeric_uni_eda.xlsx"
    QUANTILE_LEVELS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.df = df

        # Directory setup
        self.reports_dir = BASE_DIR / "src" / "EDA_Summary" / "Univariate_Analysis" / "Reports" / "excel_reports"
        self.logs_dir = BASE_DIR / "src" / "EDA_Summary" / "Univariate_Analysis" / "logs"
        self.columns_json_path = BASE_DIR / "src" / "Data" / "classified_columns.json"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_log("UnivariateNUMEDA", log_dir=self.logs_dir)

    # ============================================================
    # Helpers & Column Resolution
    # ============================================================

    def load_classified_columns(self) -> dict:
        try:
            with open(self.columns_json_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            self.logger.error(f"Failed to load classified columns from {self.columns_json_path}: {e}")
            raise CustomException(e, self.logger)

    def get_numeric_columns(self) -> list[str]:
        try:
            classified = self.load_classified_columns()
            return list(classified["univariate_candidates"]["numeric"])
        except Exception as e:
            self.logger.warning(f"Could not load columns from JSON ({e}). Falling back to DataFrame dtypes.")
            if self.df is not None:
                cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                return [c for c in cols if not pd.api.types.is_bool_dtype(self.df[c])]
            raise CustomException("DataFrame must be provided when fallback detection triggers.", self.logger)

    @staticmethod
    def _get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
        series = pd.to_numeric(df[column], errors="coerce")
        return series.replace([np.inf, -np.inf], np.nan).dropna()

    def _safely_process_columns(
        self, num_cols: list[str], process_fn: Callable[[str], dict[str, Any] | list[dict[str, Any]] | None], metric_name: str
    ) -> pd.DataFrame:
        """Utility wrapper to safely apply a calculation per column and catch exceptions."""
        results = []
        for col in num_cols:
            try:
                res = process_fn(col)
                if res is not None:
                    if isinstance(res, list):
                        results.extend(res)
                    else:
                        results.append(res)
                self.logger.info(f"Computed {metric_name} for: {col}")
            except Exception as e:
                self.logger.warning(f"Error computing {metric_name} for {col}: {e}")
        return pd.DataFrame(results)

    # ============================================================
    # Calculations & Computations
    # ============================================================

    def generate_basic_statistics(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info(f"Computing basic statistics for {len(num_cols)} numeric features...")
        total_rows = len(self.df)

        def _calc(col: str) -> dict[str, Any]:
            raw = pd.to_numeric(self.df[col], errors="coerce")
            valid = raw.replace([np.inf, -np.inf], np.nan).dropna()
            v_count = len(valid)
            m_count = total_rows - v_count
            u_count = valid.nunique(dropna=True)
            return {
                "feature": col,
                "dtype": str(self.df[col].dtype),
                "total_rows": total_rows,
                "valid_count": v_count,
                "missing_count": m_count,
                "missing_percentage": (m_count / total_rows * 100) if total_rows else 0.0,
                "unique_count": u_count,
                "unique_percentage": (u_count / v_count * 100) if v_count else 0.0,
            }

        return self._safely_process_columns(num_cols, _calc, "basic statistics")

    def generate_central_tendency(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing central tendency metrics...")

        def _calc(col: str) -> dict[str, Any] | None:
            series = self._get_numeric_series(self.df, col)
            if series.empty:
                return None
            mode_vals = series.mode()
            mode = mode_vals.iloc[0] if not mode_vals.empty else np.nan
            mode_count = int((series == mode).sum()) if not pd.isna(mode) else 0
            return {
                "feature": col,
                "mean": series.mean(),
                "median": series.median(),
                "mode": mode,
                "mode_count": mode_count,
                "mode_percentage": (mode_count / len(series) * 100),
            }

        return self._safely_process_columns(num_cols, _calc, "central tendency")

    def generate_dispersion_statistics(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing dispersion statistics...")

        def _calc(col: str) -> dict[str, Any] | None:
            series = self._get_numeric_series(self.df, col)
            if series.empty:
                return None
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            mean, std = series.mean(), series.std()
            return {
                "feature": col,
                "variance": series.var(),
                "standard_deviation": std,
                "minimum": series.min(),
                "maximum": series.max(),
                "range": series.max() - series.min(),
                "iqr": q3 - q1,
                "mad": np.mean(np.abs(series - mean)),
                "coefficient_of_variation": (std / abs(mean)) if not np.isclose(mean, 0) else np.nan,
            }

        return self._safely_process_columns(num_cols, _calc, "dispersion statistics")

    def generate_quantiles(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing quantiles...")

        def _calc(col: str) -> dict[str, Any] | None:
            series = self._get_numeric_series(self.df, col)
            if series.empty:
                return None
            v = series.quantile(self.QUANTILE_LEVELS)
            return {
                "feature": col,
                "q0": v.loc[0.0],
                "q01": v.loc[0.01],
                "q05": v.loc[0.05],
                "q10": v.loc[0.10],
                "q25": v.loc[0.25],
                "q50": v.loc[0.50],
                "q75": v.loc[0.75],
                "q90": v.loc[0.90],
                "q95": v.loc[0.95],
                "q99": v.loc[0.99],
                "q100": v.loc[1.0],
            }

        return self._safely_process_columns(num_cols, _calc, "quantiles")

    def generate_distribution_shape(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing distribution shapes...")

        def _calc(col: str) -> dict[str, Any] | None:
            series = self._get_numeric_series(self.df, col)
            if len(series) < 3:
                return None
            mean, median = series.mean(), series.median()
            return {
                "feature": col,
                "skewness": skew(series, bias=False),
                "kurtosis": kurtosis(series, fisher=True, bias=False),
                "mean": mean,
                "median": median,
                "mean_median_difference": mean - median,
                "mean_median_ratio": (mean / median) if not np.isclose(median, 0) else np.nan,
            }

        return self._safely_process_columns(num_cols, _calc, "distribution shape")

    def generate_missing_values(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing missing value metrics...")
        total_rows = len(self.df)

        def _calc(col: str) -> dict[str, Any]:
            raw = pd.to_numeric(self.df[col], errors="coerce")
            inf_count = int(np.isinf(raw.values).sum())
            miss_count = int(raw.isna().sum())
            return {
                "feature": col,
                "total_rows": total_rows,
                "missing_count": miss_count,
                "missing_percentage": (miss_count / total_rows * 100) if total_rows else 0.0,
                "infinite_count": inf_count,
                "valid_count": total_rows - miss_count - inf_count,
            }

        return self._safely_process_columns(num_cols, _calc, "missing values")

    def generate_zero_values(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing zero value metrics...")

        def _calc(col: str) -> dict[str, Any]:
            series = self._get_numeric_series(self.df, col)
            z_count = int((series == 0).sum())
            return {
                "feature": col,
                "valid_count": len(series),
                "zero_count": z_count,
                "zero_percentage": (z_count / len(series) * 100) if len(series) else 0.0,
            }

        return self._safely_process_columns(num_cols, _calc, "zero values")

    def generate_negative_values(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing negative value metrics...")

        def _calc(col: str) -> dict[str, Any]:
            series = self._get_numeric_series(self.df, col)
            neg_count = int((series < 0).sum())
            return {
                "feature": col,
                "valid_count": len(series),
                "negative_count": neg_count,
                "negative_percentage": (neg_count / len(series) * 100) if len(series) else 0.0,
                "minimum_value": series.min() if not series.empty else np.nan,
            }

        return self._safely_process_columns(num_cols, _calc, "negative values")

    def generate_unique_values(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing unique value metrics...")

        def _calc(col: str) -> dict[str, Any]:
            series = self._get_numeric_series(self.df, col)
            v_count, u_count = len(series), series.nunique()
            return {
                "feature": col,
                "valid_count": v_count,
                "unique_count": u_count,
                "unique_percentage": (u_count / v_count * 100) if v_count else 0.0,
                "duplicate_count": v_count - u_count,
                "constant": u_count <= 1,
                "low_cardinality": u_count <= self.LOW_CARDINALITY_THRESHOLD,
            }

        return self._safely_process_columns(num_cols, _calc, "unique values")

    def generate_outlier_statistics(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing outlier statistics...")

        def _calc(col: str) -> dict[str, Any] | None:
            series = self._get_numeric_series(self.df, col)
            if series.empty:
                return None
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            lower_out = int((series < lower).sum())
            upper_out = int((series > upper).sum())
            out_count = lower_out + upper_out
            return {
                "feature": col,
                "valid_count": len(series),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "outlier_count": out_count,
                "outlier_percentage": (out_count / len(series) * 100),
                "lower_outlier_count": lower_out,
                "upper_outlier_count": upper_out,
            }

        return self._safely_process_columns(num_cols, _calc, "outlier statistics")

    def generate_outlier_bounds(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing outlier bounds...")

        def _calc(col: str) -> dict[str, Any] | None:
            series = self._get_numeric_series(self.df, col)
            if series.empty:
                return None
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "lower_outlier_count": int((series < lower).sum()),
                "upper_outlier_count": int((series > upper).sum()),
            }

        return self._safely_process_columns(num_cols, _calc, "outlier bounds")

    def generate_histogram_distribution(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing histogram distributions...")

        def _calc(col: str) -> list[dict[str, Any]] | None:
            series = self._get_numeric_series(self.df, col)
            if len(series) < 2:
                return None

            if series.nunique() == 1:
                val = series.iloc[0]
                return [{
                    "feature": col,
                    "bin_start": val,
                    "bin_end": val,
                    "bin_midpoint": val,
                    "count": len(series),
                    "percentage": 100.0,
                }]

            counts, edges = np.histogram(series.to_numpy(), bins=self.HISTOGRAM_BINS)
            total = counts.sum()
            return [
                {
                    "feature": col,
                    "bin_start": edges[idx],
                    "bin_end": edges[idx + 1],
                    "bin_midpoint": (edges[idx] + edges[idx + 1]) / 2,
                    "count": int(count),
                    "percentage": (count / total * 100) if total else 0.0,
                }
                for idx, count in enumerate(counts)
            ]

        return self._safely_process_columns(num_cols, _calc, "histogram distribution")

    def generate_distribution_diagnostics(self, num_cols: list[str]) -> pd.DataFrame:
        self.logger.info("Computing distribution diagnostics...")

        def _calc(col: str) -> dict[str, Any] | None:
            series = self._get_numeric_series(self.df, col)
            n = len(series)
            if n < 3:
                return None

            norm_stat, norm_p, and_stat = np.nan, np.nan, np.nan
            if n >= 8 and series.nunique() > 1:
                try:
                    res = normaltest(series)
                    norm_stat, norm_p = res.statistic, res.pvalue
                except Exception as exc:
                    self.logger.warning(f"Normality test failed for {col}: {exc}")

            if series.nunique() > 1:
                try:
                    and_stat = anderson(series, dist="norm").statistic
                except Exception as exc:
                    self.logger.warning(f"Anderson test failed for {col}: {exc}")

            return {
                "feature": col,
                "n": n,
                "normaltest": "dagostino_pearson" if n >= 8 else None,
                "normaltest_statistic": norm_stat,
                "normaltest_p_value": norm_p,
                "anderson_darling_statistic": and_stat,
            }

        return self._safely_process_columns(num_cols, _calc, "distribution diagnostics")

    # ============================================================
    # Saving & Pipeline Execution
    # ============================================================

    def save_workbook(self, results: dict[str, pd.DataFrame]) -> None:
        try:
            wb_path = self.reports_dir / self.WORKBOOK_NAME
            with pd.ExcelWriter(wb_path, engine="openpyxl") as writer:
                for sheet, df in results.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)
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

        self.logger.info("Starting NUMERIC-UNI exploratory data analysis...")
        num_cols = self.get_numeric_columns()

        # Dynamic generation map
        metrics_map = {
            "basic_statistics": self.generate_basic_statistics,
            "central_tendency": self.generate_central_tendency,
            "dispersion_statistics": self.generate_dispersion_statistics,
            "quantiles": self.generate_quantiles,
            "distribution_shape": self.generate_distribution_shape,
            "missing_values": self.generate_missing_values,
            "zero_values": self.generate_zero_values,
            "negative_values": self.generate_negative_values,
            "unique_values": self.generate_unique_values,
            "outlier_statistics": self.generate_outlier_statistics,
            "outlier_bounds": self.generate_outlier_bounds,
            "histogram_distribution": self.generate_histogram_distribution,
            "distribution_diagnostics": self.generate_distribution_diagnostics,
        }

        results = {sheet: func(num_cols) for sheet, func in metrics_map.items()}
        self.save_workbook(results)
        self.logger.info("NUMERIC-UNI EDA pipeline completed successfully.")

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

def uni_num_stats(file_path: Path | str | None = None) -> None:
    """Wrapper function to instantiate and run numeric univariate statistics."""
    if file_path is None:
        file_path = BASE_DIR / "src" / "Data" / "cleaned_ecommerce_dataset.csv"

    eda_engine = NumericUnivariateEDA()
    eda_engine.run_from_file(file_path=file_path)


if __name__ == "__main__":
    uni_num_stats()