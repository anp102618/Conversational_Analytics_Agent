import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import f, kruskal, mannwhitneyu, ttest_ind

from src.config import BASE_DIR
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class NumericCategoricalEDA:
    """Numeric-categorical bivariate EDA engine."""

    WORKBOOK_NAME = "num_cat_eda.xlsx"
    MIN_SAMPLE_SIZE = 2
    POST_HOC_ALPHA = 0.05

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.df = df
        self.reports_dir = BASE_DIR / "src/EDA_Summary/Bivariate_Analysis/Reports/excel_reports"
        self.logs_dir = BASE_DIR / "src/EDA_Summary/Bivariate_Analysis/logs"
        self.columns_json_path = BASE_DIR / "src/Data/classified_columns.json"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_log("BivariateNUMCATEDA", log_dir=self.logs_dir)

    # ============================================================
    # Column Resolution & Pair Helpers
    # ============================================================
    @track_performance
    def load_classified_columns(self) -> dict[str, Any]:
        """Load classified column metadata."""
        try:
            with self.columns_json_path.open("r", encoding="utf-8") as f_in:
                return json.load(f_in)
        except Exception as exc:
            self.logger.error("Failed to load classified columns: %s", exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def _get_columns(self, key: str, fallback_types: list[Any], exclude_bool: bool = False) -> list[str]:
        """Resolve classified columns with dtype fallback."""
        try:
            columns = self.load_classified_columns()["bivariate_candidates"][key]
            if not isinstance(columns, list):
                raise ValueError(f"{key!r} must be a list.")
            valid = [c for c in columns if self.df is not None and c in self.df.columns]
            if valid:
                return valid
            raise ValueError(f"No classified {key} columns exist.")
        except Exception as exc:
            self.logger.warning("Could not load %s from classification (%s). Using fallback.", key, exc)
            if self.df is None:
                raise CustomException("DataFrame is required for column fallback.", self.logger) from exc
            cols = self.df.select_dtypes(include=fallback_types).columns.tolist()
            return [c for c in cols if not pd.api.types.is_bool_dtype(self.df[c])] if exclude_bool else cols

    @track_performance
    def get_numeric_columns(self) -> list[str]:
        return self._get_columns("numeric", [np.number], exclude_bool=True)

    @track_performance
    def get_categorical_columns(self) -> list[str]:
        return self._get_columns("categorical", ["object", "category", "string"])

    @track_performance
    def _get_valid_pair(self, num: str, cat: str) -> pd.DataFrame:
        """Return valid observations for a numeric-categorical pair."""
        pair = self.df[[num, cat]].copy()
        pair[num] = pd.to_numeric(pair[num], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return pair.dropna(subset=[num, cat])

    @track_performance
    def _get_groups(self, num: str, cat: str) -> dict[str, np.ndarray]:
        """Group numeric values by category."""
        pair = self._get_valid_pair(num, cat)
        return {
            str(c): g[num].to_numpy(dtype=float)
            for c, g in pair.groupby(cat, observed=True) if len(g)
        }

    @track_performance
    def _get_filtered_groups(self, num: str, cat: str, min_size: int | None = None) -> dict[str, np.ndarray]:
        """Helper to get groups matching a minimum sample size requirement."""
        min_size = self.MIN_SAMPLE_SIZE if min_size is None else min_size
        return {k: v for k, v in self._get_groups(num, cat).items() if len(v) >= min_size}

    @track_performance
    def _safely_process_pairs(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        process_fn: Callable[[str, str], list[dict[str, Any]] | dict[str, Any] | None],
        metric_name: str,
    ) -> pd.DataFrame:
        """Safely process every numeric-categorical pair."""
        results: list[dict[str, Any]] = []
        for num in numeric_columns:
            for cat in categorical_columns:
                try:
                    res = process_fn(num, cat)
                    if res is None:
                        continue
                    results.extend(res if isinstance(res, list) else [res])
                    self.logger.info("Computed %s for (%s, %s)", metric_name, num, cat)
                except Exception as exc:
                    self.logger.warning("Error computing %s for (%s, %s): %s", metric_name, num, cat, exc)
        return pd.DataFrame(results)

    # ============================================================
    # Group Statistics & Two-Group Helpers
    # ============================================================
    @track_performance
    def generate_group_statistics(self, numeric_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
        """Generate descriptive statistics by category."""
        def calculate(num: str, cat: str) -> list[dict[str, Any]]:
            pair = self._get_valid_pair(num, cat)
            res = []
            for category, group in pair.groupby(cat, observed=True):
                vals = group[num]
                if vals.empty:
                    continue
                q25, q75 = vals.quantile([0.25, 0.75])
                res.append({
                    "numeric_feature": num, "categorical_feature": cat, "category": str(category),
                    "sample_size": len(vals), "mean": vals.mean(), "median": vals.median(),
                    "std": vals.std(), "variance": vals.var(), "minimum": vals.min(),
                    "q25": q25, "q75": q75, "maximum": vals.max(), "iqr": q75 - q25,
                })
            return res

        return self._safely_process_pairs(numeric_columns, categorical_columns, calculate, "group statistics")

    @track_performance
    def _two_groups(self, num: str, cat: str) -> tuple[dict[str, np.ndarray], list[str]] | None:
        groups = self._get_groups(num, cat)
        names = list(groups)
        if len(groups) != 2 or any(len(groups[n]) < self.MIN_SAMPLE_SIZE for n in names):
            return None
        return groups, names

    @track_performance
    def generate_welch_t_test(self, numeric_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
        """Generate Welch independent two-sample t-tests."""
        def calculate(num: str, cat: str) -> dict[str, Any] | None:
            res = self._two_groups(num, cat)
            if res is None:
                return None
            groups, (n1, n2) = res
            g1, g2 = groups[n1], groups[n2]
            if np.var(g1, ddof=1) == 0 and np.var(g2, ddof=1) == 0:
                return None
            stat, p_val = ttest_ind(g1, g2, equal_var=False)
            return {
                "numeric_feature": num, "categorical_feature": cat, "group_1": n1, "group_2": n2,
                "n_group_1": len(g1), "n_group_2": len(g2), "mean_group_1": np.mean(g1),
                "mean_group_2": np.mean(g2), "welch_t_statistic": stat, "p_value": p_val,
            }

        return self._safely_process_pairs(numeric_columns, categorical_columns, calculate, "Welch t-test")

    @track_performance
    def generate_mann_whitney_u(self, numeric_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
        """Generate two-group Mann-Whitney U tests."""
        def calculate(num: str, cat: str) -> dict[str, Any] | None:
            res = self._two_groups(num, cat)
            if res is None:
                return None
            groups, (n1, n2) = res
            g1, g2 = groups[n1], groups[n2]
            stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
            return {
                "numeric_feature": num, "categorical_feature": cat, "group_1": n1, "group_2": n2,
                "n_group_1": len(g1), "n_group_2": len(g2), "median_group_1": np.median(g1),
                "median_group_2": np.median(g2), "u_statistic": stat, "p_value": p_val,
            }

        return self._safely_process_pairs(numeric_columns, categorical_columns, calculate, "Mann-Whitney U")

    # ============================================================
    # Multi-Group Tests & Post-Hoc
    # ============================================================
    @track_performance
    @staticmethod
    def _welch_anova(groups: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
        """Calculate Welch's ANOVA."""
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        k = len(groups)
        if k < 3:
            raise ValueError("Welch ANOVA requires at least three groups.")

        n = np.array([len(x) for x in groups.values()], dtype=float)
        means = np.array([np.mean(x) for x in groups.values()])
        vars_ = np.array([np.var(x, ddof=1) for x in groups.values()])
        vars_ = np.where(vars_ == 0, np.finfo(float).eps, vars_)

        weights = n / vars_
        w_sum = weights.sum()
        w_mean = np.sum(weights * means) / w_sum

        numerator = np.sum(weights * (means - w_mean) ** 2) / (k - 1)
        corr_term = np.sum((1 / (n - 1)) * (1 - weights / w_sum) ** 2)
        corr = 1 + (2 * (k - 2) / (k**2 - 1)) * corr_term

        stat = numerator / corr
        df1, df2 = k - 1, (k**2 - 1) / (3 * corr_term) if corr_term > 0 else np.inf
        return stat, f.sf(stat, df1, df2), df1, df2

    @track_performance
    def generate_welch_anova(self, numeric_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
        """Generate Welch ANOVA for 3+ groups."""
        def calculate(num: str, cat: str) -> dict[str, Any] | None:
            groups = self._get_filtered_groups(num, cat, min_size=2)
            if len(groups) < 3:
                return None
            stat, p_val, df1, df2 = self._welch_anova(groups)
            return {
                "numeric_feature": num, "categorical_feature": cat, "number_of_groups": len(groups),
                "groups": " | ".join(groups), "welch_f_statistic": stat, "df1": df1, "df2": df2, "p_value": p_val,
            }

        return self._safely_process_pairs(numeric_columns, categorical_columns, calculate, "Welch ANOVA")

    @track_performance
    def generate_kruskal_wallis(self, numeric_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
        """Generate Kruskal-Wallis tests."""
        def calculate(num: str, cat: str) -> dict[str, Any] | None:
            groups = self._get_filtered_groups(num, cat)
            if len(groups) < 3:
                return None
            stat, p_val = kruskal(*groups.values())
            return {
                "numeric_feature": num, "categorical_feature": cat, "number_of_groups": len(groups),
                "groups": " | ".join(groups), "kruskal_h_statistic": stat, "p_value": p_val,
            }

        return self._safely_process_pairs(numeric_columns, categorical_columns, calculate, "Kruskal-Wallis")

    @track_performance
    def generate_post_hoc(
        self, numeric_columns: list[str], categorical_columns: list[str], alpha: float | None = None
    ) -> pd.DataFrame:
        """Pairwise Mann-Whitney tests with Bonferroni correction."""
        alpha = self.POST_HOC_ALPHA if alpha is None else alpha

        def calculate(num: str, cat: str) -> list[dict[str, Any]] | None:
            groups = self._get_filtered_groups(num, cat)
            if len(groups) < 3:
                return None

            pairs = list(combinations(groups, 2))
            comps = len(pairs)
            res = []
            for g1_name, g2_name in pairs:
                g1, g2 = groups[g1_name], groups[g2_name]
                stat, raw_p = mannwhitneyu(g1, g2, alternative="two-sided")
                adj_p = min(raw_p * comps, 1.0)
                res.append({
                    "numeric_feature": num, "categorical_feature": cat,
                    "group_1": g1_name, "group_2": g2_name, "n_group_1": len(g1), "n_group_2": len(g2),
                    "u_statistic": stat, "raw_p_value": raw_p,
                    "bonferroni_adjusted_p_value": adj_p, "significant": adj_p < alpha,
                })
            return res

        return self._safely_process_pairs(numeric_columns, categorical_columns, calculate, "post-hoc analysis")

    # ============================================================
    # Effect Sizes
    # ============================================================
    @track_performance
    @staticmethod
    def _calculate_cohens_d(group_1: np.ndarray, group_2: np.ndarray) -> float:
        """Calculate Cohen's d."""
        n1, n2 = len(group_1), len(group_2)
        v1, v2 = np.var(group_1, ddof=1), np.var(group_2, ddof=1)
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        return np.nan if pooled <= 0 else (np.mean(group_1) - np.mean(group_2)) / np.sqrt(pooled)

    @track_performance
    @staticmethod
    def _calculate_eta_squared(groups: dict[str, np.ndarray]) -> float:
        """Calculate eta-squared."""
        values = np.concatenate(list(groups.values()))
        g_mean = np.mean(values)
        between_ss = sum(len(g) * (np.mean(g) - g_mean) ** 2 for g in groups.values())
        total_ss = np.sum((values - g_mean) ** 2)
        return np.nan if total_ss <= 0 else between_ss / total_ss

    @track_performance
    def generate_effect_size(self, numeric_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
        """Generate Cohen's d for two groups or eta-squared for 3+."""
        def calculate(num: str, cat: str) -> dict[str, Any] | None:
            groups = self._get_filtered_groups(num, cat)
            n_groups = len(groups)

            if n_groups == 2:
                n1, n2 = list(groups)
                effect = self._calculate_cohens_d(groups[n1], groups[n2])
                return {
                    "numeric_feature": num, "categorical_feature": cat, "effect_size_type": "cohens_d",
                    "group_1": n1, "group_2": n2, "number_of_groups": 2, "effect_size": effect,
                    "absolute_effect_size": abs(effect) if not np.isnan(effect) else np.nan,
                }
            if n_groups >= 3:
                effect = self._calculate_eta_squared(groups)
                return {
                    "numeric_feature": num, "categorical_feature": cat, "effect_size_type": "eta_squared",
                    "group_1": None, "group_2": None, "number_of_groups": n_groups, "effect_size": effect,
                    "absolute_effect_size": effect,
                }
            return None

        return self._safely_process_pairs(numeric_columns, categorical_columns, calculate, "effect size")

    # ============================================================
    # Workbook / Pipeline
    # ============================================================
    @track_performance
    def save_workbook(self, results: dict[str, pd.DataFrame]) -> None:
        """Save all results to Excel."""
        try:
            path = self.reports_dir / self.WORKBOOK_NAME
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for sheet, df_sheet in results.items():
                    df_sheet.to_excel(writer, sheet_name=sheet, index=False)
            self.logger.info("Workbook created at %s (%d sheets)", path, len(results))
        except Exception as exc:
            self.logger.error("Failed to save workbook: %s", exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def run_eda(self, df: pd.DataFrame | None = None) -> None:
        """Run complete NUM-CAT EDA pipeline."""
        if df is not None:
            self.df = df

        if self.df is None or self.df.empty:
            raise CustomException("DataFrame is empty. Please provide a valid pandas DataFrame.", self.logger)

        self.logger.info("Starting NUMERIC-CATEGORICAL EDA...")
        num_cols, cat_cols = self.get_numeric_columns(), self.get_categorical_columns()

        if not num_cols:
            raise CustomException("No numeric columns available for NUM-CAT analysis.", self.logger)
        if not cat_cols:
            raise CustomException("No categorical columns available for NUM-CAT analysis.", self.logger)

        self.logger.info("Resolved %d numeric and %d categorical columns.", len(num_cols), len(cat_cols))

        methods = {
            "group_statistics": self.generate_group_statistics,
            "welch_t_test": self.generate_welch_t_test,
            "mann_whitney_u": self.generate_mann_whitney_u,
            "welch_anova": self.generate_welch_anova,
            "kruskal_wallis": self.generate_kruskal_wallis,
            #"post_hoc": self.generate_post_hoc,
            "effect_size": self.generate_effect_size,
        }

        results = {name: method(num_cols, cat_cols) for name, method in methods.items()}
        self.save_workbook(results)
        self.logger.info("NUMERIC-CATEGORICAL EDA completed successfully.")

    @track_performance
    def run_from_file(self, file_path: Path | str) -> None:
        """Load CSV and execute NUM-CAT EDA."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise CustomException(FileNotFoundError(f"The file {file_path} does not exist."), self.logger)

        try:
            self.df = pd.read_csv(file_path)
            self.logger.info("Dataset loaded with shape: %s", self.df.shape)
            self.run_eda()
        except pd.errors.EmptyDataError as exc:
            self.logger.error("Dataset is empty: %s", file_path)
            raise CustomException(exc, self.logger) from exc
        except Exception as exc:
            self.logger.error("Error processing CSV: %s", exc)
            raise CustomException(exc, self.logger) from exc

@track_performance
def num_cat_stats(file_path: Path | str | None = None) -> None:
    """Execute numeric-categorical bivariate EDA."""
    file_path = file_path or (BASE_DIR / "src/Data/cleaned_ecommerce_dataset.csv")
    NumericCategoricalEDA().run_from_file(file_path)


if __name__ == "__main__":
    num_cat_stats()