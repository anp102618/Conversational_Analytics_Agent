#====================================================
# Categorical Univariate Statistics
#====================================================


"""
Categorical Univariate Statistics
=================================
Deterministic categorical univariate analysis tools for the Pandas
Data Agent.
Supported analyses
------------------
- Basic categorical statistics
- Frequency distribution
- Central tendency (mode)
- Cardinality and diversity
- Missing values
- Rare categories
- Top categories and cumulative coverage
- Category imbalance and concentration
- Duplicate observations
- Category data-quality diagnostics
Design principles
-----------------
1. Pandas performs all deterministic calculations.
2. The LLM selects tools and supplies analytical parameters.
3. The DataFrame remains internal to the engine.
4. Every tool validates the requested column.
5. Tool results use JSON-compatible structured dictionaries.
6. Project logging and CustomException are used consistently.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class CategoricalUnivariateStatistics:
    """
    Deterministic categorical univariate analysis engine.
    Parameters
    ----------
    df:
        Pandas DataFrame containing the dataset.
    logger_name:
        Logger name used by the application.
    Notes
    -----
    The engine does not modify the source DataFrame or write files.
    It is designed for dynamic analytical queries from a Pandas
    Data Agent.
    """
    DEFAULT_TOP_N = 10
    DEFAULT_RARE_THRESHOLD = 5.0
    LOW_CARDINALITY_THRESHOLD = 10

    def __init__(self, df: pd.DataFrame, logger_name: str = "PandasAgentCategoricalUNI") -> None:
        """Initialize the categorical analysis engine."""
        if df is None:
            raise CustomException("DataFrame cannot be None.", get_log(logger_name))
        if df.empty:
            raise CustomException("DataFrame cannot be empty.", get_log(logger_name))
        self.df = df
        self.logger = get_log(logger_name)
        self.logger.info(
            "Initialized categorical statistics engine with %d rows and %d columns.",
            df.shape[0], df.shape[1],
        )

    def _validate_column(self, column: str) -> pd.Series:
        """
        Validate a requested column.
        Categorical analysis can be applied to object, string,
        category, boolean, or numeric-coded columns. The agent
        determines whether a column should be interpreted as
        categorical.
        """
        try:
            if not isinstance(column, str) or not column.strip():
                raise ValueError("Column name must be a non-empty string.")
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' does not exist.")
            return self.df[column]
        except Exception as exc:
            self.logger.error("Categorical column validation failed for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @staticmethod
    def _safe_percentage(numerator: int | float, denominator: int | float) -> float:
        """Calculate a percentage safely."""
        if denominator == 0:
            return 0.0
        return round(float(numerator) / float(denominator) * 100, 4)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert common Pandas and NumPy values to JSON-safe values."""
        if value is None or value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _category_label(value: Any) -> str:
        """Create a stable string label for category output."""
        safe_value = CategoricalUnivariateStatistics._json_safe(value)
        return "<MISSING>" if safe_value is None else str(safe_value)

    def _get_series(self, column: str) -> pd.Series:
        """
        Return the original categorical series after validation.
        Missing values are not removed here because different
        analyses need to report them explicitly.
        """
        return self._validate_column(column)

    def _get_valid_series(self, column: str) -> pd.Series:
        """Return non-missing categorical observations."""
        return self._get_series(column).dropna()

    def _result(self, tool_name: str, column: str, **values: Any) -> dict[str, Any]:
        """Build a standardized JSON-compatible response."""
        result = {
            "status": "success",
            "tool": tool_name,
            "analysis_type": "univariate",
            "data_type": "categorical",
            "column": column,
        }
        for key, value in values.items():
            if isinstance(value, dict):
                result[key] = {str(k): self._json_safe(v) for k, v in value.items()}
            elif isinstance(value, list):
                result[key] = [
                    {str(k): self._json_safe(v) for k, v in item.items()}
                    if isinstance(item, dict) else self._json_safe(item)
                    for item in value
                ]
            else:
                result[key] = self._json_safe(value)
        return result

    def _frequency_table(self, column: str, include_missing: bool = False) -> pd.Series:
        """
        Return category frequencies.
        dropna=False includes missing observations as a category
        for frequency analysis.
        """
        series = self._get_series(column)
        return series.value_counts(dropna=not include_missing, sort=True)

    @track_performance
    def categorical_statistics(self, column: str) -> dict[str, Any]:
        """
        Calculate basic categorical statistics.
        Includes:
        - Total rows
        - Valid and missing counts
        - Missing percentage
        - Unique category count
        - Unique percentage
        - Most frequent category
        - Most frequent category percentage
        - Original dtype
        """
        try:
            series = self._get_series(column)
            total_rows = len(series)
            missing_count = int(series.isna().sum())
            valid = series.dropna()
            valid_count = len(valid)
            unique_count = int(valid.nunique())
            frequencies = valid.value_counts()
            if frequencies.empty:
                mode, mode_count = None, 0
            else:
                mode = frequencies.index[0]
                mode_count = int(frequencies.iloc[0])
            self.logger.info("Computed categorical statistics for '%s'.", column)
            return self._result(
                "categorical_statistics", column,
                dtype=str(series.dtype),
                total_rows=total_rows,
                valid_count=valid_count,
                missing_count=missing_count,
                missing_percentage=self._safe_percentage(missing_count, total_rows),
                unique_count=unique_count,
                unique_percentage=self._safe_percentage(unique_count, valid_count),
                mode=self._category_label(mode) if mode is not None else None,
                mode_count=mode_count,
                mode_percentage=self._safe_percentage(mode_count, valid_count),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed categorical statistics for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def frequency_distribution(self, column: str, include_missing: bool = False) -> dict[str, Any]:
        """
        Calculate category counts and percentages.
        Parameters
        ----------
        column:
            Categorical column to analyze.
        include_missing:
            Whether missing values should appear as a category.
        Notes
        -----
        Percentages use the total number of observations in the
        selected frequency table.
        """
        try:
            frequencies = self._frequency_table(column, include_missing=include_missing)
            denominator = int(frequencies.sum())
            distribution = [
                {
                    "category": self._category_label(category),
                    "count": int(count),
                    "percentage": self._safe_percentage(count, denominator),
                    "is_missing": bool(pd.isna(category)),
                }
                for category, count in frequencies.items()
            ]
            self.logger.info("Computed frequency distribution for '%s'.", column)
            return self._result(
                "frequency_distribution", column,
                include_missing=include_missing,
                total_count=denominator,
                category_count=len(distribution),
                distribution=distribution,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed frequency distribution for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def central_tendency(self, column: str) -> dict[str, Any]:
        """
        Calculate categorical central tendency.
        Reports:
        - Mode
        - Mode count
        - Mode percentage
        - Number of tied modes
        - All tied modes
        Categorical variables do not have a meaningful arithmetic
        mean or median.
        """
        try:
            series = self._get_valid_series(column)
            if series.empty:
                raise ValueError(f"Column '{column}' has no valid observations.")
            frequencies = series.value_counts()
            max_count = int(frequencies.max())
            modes = frequencies[frequencies == max_count]
            mode_values = [self._category_label(v) for v in modes.index.tolist()]
            self.logger.info("Computed categorical central tendency for '%s'.", column)
            return self._result(
                "central_tendency", column,
                valid_count=len(series),
                mode=mode_values[0],
                mode_count=max_count,
                mode_percentage=self._safe_percentage(max_count, len(series)),
                tied_mode_count=len(mode_values),
                modes=mode_values,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed central tendency for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def cardinality_analysis(self, column: str) -> dict[str, Any]:
        """
        Analyze categorical cardinality and diversity.
        Includes:
        - Unique categories
        - Cardinality ratio
        - Low-cardinality indicator
        - Constant-column indicator
        - Shannon entropy
        - Normalized entropy
        - Simpson concentration
        Entropy and concentration are descriptive measures of
        category distribution, not measures of data quality.
        """
        try:
            series = self._get_valid_series(column)
            if series.empty:
                raise ValueError(f"Column '{column}' has no valid observations.")
            frequencies = series.value_counts()
            n, k = len(series), len(frequencies)
            probabilities = frequencies.to_numpy(dtype=float) / n
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))
            max_entropy = float(np.log2(k)) if k > 1 else 0.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            simpson_concentration = float(np.sum(probabilities ** 2))
            return self._result(
                "cardinality_analysis", column,
                valid_count=n,
                unique_count=k,
                cardinality_ratio=round(k / n, 6),
                unique_percentage=self._safe_percentage(k, n),
                low_cardinality=k <= self.LOW_CARDINALITY_THRESHOLD,
                constant=k <= 1,
                shannon_entropy=round(entropy, 6),
                maximum_entropy=round(max_entropy, 6),
                normalized_entropy=round(normalized_entropy, 6),
                simpson_concentration=round(simpson_concentration, 6),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed cardinality analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def missing_values(self, column: str) -> dict[str, Any]:
        """Analyze missing and non-missing categorical observations."""
        try:
            series = self._get_series(column)
            total_rows = len(series)
            missing_count = int(series.isna().sum())
            valid_count = total_rows - missing_count
            return self._result(
                "missing_values", column,
                total_rows=total_rows,
                missing_count=missing_count,
                valid_count=valid_count,
                missing_percentage=self._safe_percentage(missing_count, total_rows),
                valid_percentage=self._safe_percentage(valid_count, total_rows),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed missing-value analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def rare_categories(
        self,
        column: str,
        threshold_percentage: float = DEFAULT_RARE_THRESHOLD,
        include_missing: bool = False,
    ) -> dict[str, Any]:
        """
        Identify categories below a frequency percentage threshold.
        Parameters
        ----------
        threshold_percentage:
            A category is rare when its percentage is strictly
            below this threshold.
        include_missing:
            Whether missing observations should be considered
            as a category.
        """
        try:
            if not 0 <= threshold_percentage <= 100:
                raise ValueError("threshold_percentage must be between 0 and 100.")
            frequencies = self._frequency_table(column, include_missing=include_missing)
            total = int(frequencies.sum())
            rare = [
                {
                    "category": self._category_label(category),
                    "count": int(count),
                    "percentage": pct,
                    "is_missing": bool(pd.isna(category)),
                }
                for category, count in frequencies.items()
                if (pct := self._safe_percentage(count, total)) < threshold_percentage
            ]
            rare.sort(key=lambda item: item["percentage"])
            return self._result(
                "rare_categories", column,
                threshold_percentage=threshold_percentage,
                include_missing=include_missing,
                total_count=total,
                rare_category_count=len(rare),
                rare_observation_count=sum(item["count"] for item in rare),
                rare_categories=rare,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed rare-category analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def top_categories(self, column: str, top_n: int = DEFAULT_TOP_N) -> dict[str, Any]:
        """
        Return the most frequent categories and cumulative coverage.
        Cumulative percentage shows how much of the valid dataset
        is represented by the first N categories.
        """
        try:
            if top_n < 1:
                raise ValueError("top_n must be at least 1.")
            series = self._get_valid_series(column)
            if series.empty:
                raise ValueError(f"Column '{column}' has no valid observations.")
            frequencies = series.value_counts()
            total = int(frequencies.sum())
            distribution = []
            cumulative_count = 0
            for category, count in frequencies.head(top_n).items():
                count = int(count)
                cumulative_count += count
                distribution.append({
                    "category": self._category_label(category),
                    "count": count,
                    "percentage": self._safe_percentage(count, total),
                    "cumulative_count": cumulative_count,
                    "cumulative_percentage": self._safe_percentage(cumulative_count, total),
                })
            return self._result(
                "top_categories", column,
                top_n=top_n,
                valid_count=total,
                returned_categories=len(distribution),
                cumulative_coverage_percentage=self._safe_percentage(cumulative_count, total),
                top_categories=distribution,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed top-category analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def category_imbalance(self, column: str) -> dict[str, Any]:
        """
        Describe category concentration and imbalance.
        Includes:
        - Largest category share
        - Smallest category share
        - Largest-to-smallest frequency ratio
        - Shannon entropy
        - Normalized entropy
        - Simpson concentration
        - Effective number of categories
        This tool reports metrics without assigning an
        evaluative imbalance label.
        """
        try:
            series = self._get_valid_series(column)
            if series.empty:
                raise ValueError(f"Column '{column}' has no valid observations.")
            frequencies = series.value_counts()
            n, k = len(series), len(frequencies)
            largest_count = int(frequencies.max())
            smallest_count = int(frequencies.min())
            probabilities = frequencies.to_numpy(dtype=float) / n
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))
            max_entropy = float(np.log2(k)) if k > 1 else 0.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            concentration = float(np.sum(probabilities ** 2))
            effective_categories = 1.0 / concentration if concentration > 0 else 0.0
            return self._result(
                "category_imbalance", column,
                valid_count=n,
                category_count=k,
                largest_category=self._category_label(frequencies.index[0]),
                largest_category_count=largest_count,
                largest_category_percentage=self._safe_percentage(largest_count, n),
                smallest_category=self._category_label(frequencies.index[-1]),
                smallest_category_count=smallest_count,
                smallest_category_percentage=self._safe_percentage(smallest_count, n),
                largest_to_smallest_ratio=(
                    largest_count / smallest_count if smallest_count > 0 else None
                ),
                shannon_entropy=round(entropy, 6),
                normalized_entropy=round(normalized_entropy, 6),
                simpson_concentration=round(concentration, 6),
                effective_category_count=round(effective_categories, 6),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed category imbalance analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def duplicate_values(self, column: str) -> dict[str, Any]:
        """
        Analyze repeated categorical observations.
        Duplicate observations are valid repeated category values,
        not duplicate DataFrame rows.
        """
        try:
            series = self._get_valid_series(column)
            valid_count = len(series)
            unique_count = int(series.nunique())
            duplicate_count = valid_count - unique_count
            frequencies = series.value_counts()
            repeated = [
                {
                    "category": self._category_label(category),
                    "count": int(count),
                    "excess_occurrences": int(count - 1),
                }
                for category, count in frequencies.items()
                if count > 1
            ]
            return self._result(
                "duplicate_values", column,
                valid_count=valid_count,
                unique_count=unique_count,
                duplicate_count=duplicate_count,
                duplicate_percentage=self._safe_percentage(duplicate_count, valid_count),
                repeated_category_count=len(repeated),
                repeated_categories=repeated,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed duplicate-value analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def category_quality(self, column: str, top_n: int = DEFAULT_TOP_N) -> dict[str, Any]:
        """
        Inspect common formatting issues in string categories.
        Detects:
        - Empty strings
        - Whitespace-only strings
        - Leading/trailing whitespace
        - Mixed casing variants
        - Potential normalized-label collisions
        This is diagnostic only: the source data is never changed.
        Normalization checks apply only to string values.
        """
        try:
            if top_n < 1:
                raise ValueError("top_n must be at least 1.")
            series = self._get_series(column)
            total_rows = len(series)
            missing_count = int(series.isna().sum())
            valid = series.dropna()
            string_mask = valid.map(lambda v: isinstance(v, str))
            strings = valid[string_mask]
            empty_count = int(strings.map(lambda v: v == "").sum())
            whitespace_only_count = int(strings.map(lambda v: v.strip() == "").sum())
            padded_count = int(strings.map(lambda v: v != v.strip()).sum())
            normalized = strings.map(lambda v: v.strip().casefold())
            original_labels = strings.map(str)
            mapping: dict[str, set[str]] = {}
            for original, norm_label in zip(original_labels, normalized):
                mapping.setdefault(norm_label, set()).add(original)
            collisions = [
                {
                    "normalized_label": label,
                    "original_variants": sorted(variants),
                    "variant_count": len(variants),
                }
                for label, variants in mapping.items()
                if len(variants) > 1
            ]
            collisions.sort(key=lambda item: item["variant_count"], reverse=True)
            casing_variants: dict[str, set[str]] = {}
            for value in strings:
                casing_variants.setdefault(value.casefold(), set()).add(value)
            casing_issues = [
                {"normalized_label": label, "variants": sorted(variants)}
                for label, variants in casing_variants.items()
                if len(variants) > 1
            ]
            return self._result(
                "category_quality", column,
                total_rows=total_rows,
                missing_count=missing_count,
                string_value_count=len(strings),
                non_string_valid_count=len(valid) - len(strings),
                empty_string_count=empty_count,
                whitespace_only_count=whitespace_only_count,
                leading_trailing_whitespace_count=padded_count,
                potential_normalization_collision_count=len(collisions),
                normalization_collisions=collisions[:top_n],
                casing_variant_group_count=len(casing_issues),
                casing_variants=casing_issues[:top_n],
                normalization_policy=(
                    "strip whitespace and casefold for diagnostics "
                    "only; original values are unchanged"
                ),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed category quality analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools backed by this DataFrame.
        Each tool exposes only its analytical parameters.
        The DataFrame remains captured by this engine instance.
        """
        @tool
        def categorical_statistics(column: str) -> dict[str, Any]:
            """Calculate rows, missingness, unique categories and mode."""
            return self.categorical_statistics(column)

        @tool
        def frequency_distribution(column: str, include_missing: bool = False) -> dict[str, Any]:
            """Return category counts and percentages."""
            return self.frequency_distribution(column, include_missing)

        @tool
        def central_tendency(column: str) -> dict[str, Any]:
            """Calculate mode and all tied modes."""
            return self.central_tendency(column)

        @tool
        def cardinality_analysis(column: str) -> dict[str, Any]:
            """Analyze unique categories, entropy and diversity."""
            return self.cardinality_analysis(column)

        @tool
        def missing_values(column: str) -> dict[str, Any]:
            """Calculate categorical missing-value metrics."""
            return self.missing_values(column)

        @tool
        def rare_categories(
            column: str,
            threshold_percentage: float = self.DEFAULT_RARE_THRESHOLD,
            include_missing: bool = False,
        ) -> dict[str, Any]:
            """Identify categories below a percentage threshold."""
            return self.rare_categories(column, threshold_percentage, include_missing)

        @tool
        def top_categories(column: str, top_n: int = self.DEFAULT_TOP_N) -> dict[str, Any]:
            """Return top categories with cumulative coverage."""
            return self.top_categories(column, top_n)

        @tool
        def category_imbalance(column: str) -> dict[str, Any]:
            """Measure category concentration and distribution."""
            return self.category_imbalance(column)

        @tool
        def duplicate_values(column: str) -> dict[str, Any]:
            """Analyze repeated category observations."""
            return self.duplicate_values(column)

        @tool
        def category_quality(column: str, top_n: int = self.DEFAULT_TOP_N) -> dict[str, Any]:
            """Inspect string formatting and category-label variants."""
            return self.category_quality(column, top_n)

        tools = [
            categorical_statistics, 
            frequency_distribution, 
            central_tendency,
            cardinality_analysis, 
            missing_values, 
            rare_categories, 
            top_categories,
            category_imbalance, 
            duplicate_values, 
            category_quality,
        ]
        
        self.logger.info("Built %d categorical statistics tools.", len(tools))
        return tools

#===============================================================================================

#====================================================
# Categorical Univariate Visualization
#====================================================
"""
Categorical Univariate Visualization.
Seaborn and Matplotlib visualizations for categorical features.
Includes LangChain tools for agent-based execution.
"""
from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class CategoricalUnivariateVisualization:
    """Create univariate visualizations for categorical columns."""
    DEFAULT_TOP_N = 15
    DEFAULT_FIGSIZE = (12, 6)

    def __init__(
        self,
        df: pd.DataFrame,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("DataFrame is empty.")
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        self.df = df
        self.top_n = top_n
        self.figsize = figsize
        self.logger = get_log(__name__)

    def _validate_column(self, column: str) -> None:
        if not isinstance(column, str) or not column.strip():
            raise ValueError("column must be a non-empty string.")
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found.")

    def _get_series(self, column: str) -> pd.Series:
        self._validate_column(column)
        return self.df[column]

    def _get_frequency(
        self,
        column: str,
        top_n: int | None = None,
        include_missing: bool = False,
    ) -> pd.Series:
        series = self._get_series(column)
        limit = top_n if top_n is not None else self.top_n
        if limit < 1:
            raise ValueError("top_n must be at least 1.")
        if include_missing:
            values = series.astype("object").where(series.notna(), "<MISSING>")
            counts = values.value_counts(dropna=False)
        else:
            counts = series.dropna().value_counts()
        return counts.head(limit)

    def _create_figure(
        self, title: str, xlabel: str, ylabel: str
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax

    @staticmethod
    def _format_axis(ax: plt.Axes, rotate: int = 0) -> None:
        ax.tick_params(axis="x", labelrotation=rotate)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if rotate:
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")

    @track_performance
    def count_plot(
        self,
        column: str,
        top_n: int | None = None,
        include_missing: bool = False,
        rotate: int = 45,
    ) -> plt.Figure:
        """Plot the frequency of the most common categories."""
        try:
            counts = self._get_frequency(column, top_n, include_missing)
            if counts.empty:
                raise ValueError(f"No values available for '{column}'.")
            fig, ax = self._create_figure(f"Category Frequency: {column}", column, "Count")
            sns.barplot(x=counts.index.map(str), y=counts.values, ax=ax)
            self._format_axis(ax, rotate)
            fig.tight_layout()
            return fig
        except Exception as exc:
            self.logger.exception("Count plot failed: %s", column)
            raise CustomException(exc) from exc

    @track_performance
    def percentage_bar_plot(
        self,
        column: str,
        top_n: int | None = None,
        include_missing: bool = False,
        rotate: int = 45,
    ) -> plt.Figure:
        """Plot category percentages using the full distribution."""
        try:
            series = self._get_series(column)
            if include_missing:
                values = series.astype("object").where(series.notna(), "<MISSING>")
                all_counts = values.value_counts(dropna=False)
            else:
                all_counts = series.dropna().value_counts()
            if all_counts.empty:
                raise ValueError(f"No values available for '{column}'.")
            limit = top_n if top_n is not None else self.top_n
            if limit < 1:
                raise ValueError("top_n must be at least 1.")
            counts = all_counts.head(limit)
            percentages = counts / all_counts.sum() * 100
            fig, ax = self._create_figure(
                f"Category Percentage: {column}", column, "Percentage (%)"
            )
            sns.barplot(x=percentages.index.map(str), y=percentages.values, ax=ax)
            for i, value in enumerate(percentages.values):
                ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
            ax.set_ylim(0, max(100, percentages.max() * 1.15))
            self._format_axis(ax, rotate)
            fig.tight_layout()
            return fig
        except Exception as exc:
            self.logger.exception("Percentage plot failed: %s", column)
            raise CustomException(exc) from exc

    @track_performance
    def horizontal_count_plot(
        self,
        column: str,
        top_n: int | None = None,
        include_missing: bool = False,
    ) -> plt.Figure:
        """Plot category frequencies horizontally."""
        try:
            counts = self._get_frequency(column, top_n, include_missing)
            if counts.empty:
                raise ValueError(f"No values available for '{column}'.")
            counts = counts.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title(
                f"Category Frequency (Horizontal): {column}",
                fontsize=14, fontweight="bold",
            )
            ax.set_xlabel("Count")
            ax.set_ylabel(column)
            sns.barplot(x=counts.values, y=counts.index.map(str), ax=ax)
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            fig.tight_layout()
            return fig
        except Exception as exc:
            self.logger.exception("Horizontal count plot failed: %s", column)
            raise CustomException(exc) from exc

    @track_performance
    def pareto_chart(
        self,
        column: str,
        top_n: int | None = None,
        include_missing: bool = False,
    ) -> plt.Figure:
        """Plot descending counts and cumulative percentage."""
        try:
            series = self._get_series(column)
            if include_missing:
                values = series.astype("object").where(series.notna(), "<MISSING>")
                all_counts = values.value_counts(dropna=False)
                denominator = len(series)
            else:
                all_counts = series.dropna().value_counts()
                denominator = int(series.notna().sum())
            if all_counts.empty or denominator == 0:
                raise ValueError(f"No values available for '{column}'.")
            limit = top_n if top_n is not None else self.top_n
            if limit < 1:
                raise ValueError("top_n must be at least 1.")
            counts = all_counts.head(limit)
            cumulative = counts.cumsum() / denominator * 100
            positions = list(range(len(counts)))
            fig, ax1 = plt.subplots(figsize=self.figsize)
            ax1.set_title(f"Pareto Chart: {column}", fontsize=14, fontweight="bold")
            ax1.set_xlabel(column)
            ax1.set_ylabel("Count")
            ax1.bar(positions, counts.values)
            ax1.set_xticks(positions)
            ax1.set_xticklabels(counts.index.map(str), rotation=45, ha="right")
            ax1.grid(axis="y", alpha=0.25)
            ax1.set_axisbelow(True)
            ax2 = ax1.twinx()
            ax2.plot(positions, cumulative.values, marker="o")
            ax2.set_ylabel("Cumulative percentage (%)")
            ax2.set_ylim(0, 105)
            fig.tight_layout()
            return fig
        except Exception as exc:
            self.logger.exception("Pareto chart failed: %s", column)
            raise CustomException(exc) from exc

    @track_performance
    def pie_chart(
        self,
        column: str,
        top_n: int = 6,
        include_missing: bool = False,
    ) -> plt.Figure:
        """Plot category proportions, grouping the remainder as Other."""
        try:
            series = self._get_series(column)
            if include_missing:
                values = series.astype("object").where(series.notna(), "<MISSING>")
                counts = values.value_counts(dropna=False)
            else:
                counts = series.dropna().value_counts()
            if counts.empty:
                raise ValueError(f"No values available for '{column}'.")
            if top_n < 1:
                raise ValueError("top_n must be at least 1.")
            if len(counts) > top_n:
                shown = counts.head(top_n).copy()
                shown.loc["Other"] = counts.iloc[top_n:].sum()
            else:
                shown = counts
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title(
                f"Category Proportions: {column}", fontsize=14, fontweight="bold"
            )
            ax.pie(
                shown.values,
                labels=shown.index.map(str),
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
            )
            ax.axis("equal")
            fig.tight_layout()
            return fig
        except Exception as exc:
            self.logger.exception("Pie chart failed: %s", column)
            raise CustomException(exc) from exc

    @track_performance
    def missingness_plot(self, column: str) -> plt.Figure:
        """Plot present versus missing observation counts."""
        try:
            series = self._get_series(column)
            missing = int(series.isna().sum())
            present = int(series.notna().sum())
            total = len(series)
            labels, values = ["Present", "Missing"], [present, missing]
            fig, ax = self._create_figure(
                f"Missingness: {column}", "Value status", "Count"
            )
            sns.barplot(x=labels, y=values, ax=ax)
            for i, value in enumerate(values):
                pct = value / total * 100 if total else 0
                ax.text(i, value, f"{value:,} ({pct:.1f}%)", ha="center", va="bottom")
            ax.set_ylim(0, max(values + [1]) * 1.15)
            self._format_axis(ax)
            fig.tight_layout()
            return fig
        except Exception as exc:
            self.logger.exception("Missingness plot failed: %s", column)
            raise CustomException(exc) from exc

    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools with the DataFrame captured internally.
        The agent supplies column names and plotting options only.
        """
        visualizer = self

        @tool
        def categorical_count_plot(
            column: str,
            top_n: int = 15,
            include_missing: bool = False,
            rotate: int = 45,
        ) -> Any:
            """
            Create a categorical count plot.
            Args:
                column: Categorical column name.
                top_n: Maximum number of categories to display.
                include_missing: Include nulls as a category.
                rotate: X-axis label rotation in degrees.
            """
            return visualizer.count_plot(
                column=column, top_n=top_n, include_missing=include_missing, rotate=rotate
            )

        @tool
        def categorical_percentage_plot(
            column: str,
            top_n: int = 15,
            include_missing: bool = False,
            rotate: int = 45,
        ) -> Any:
            """
            Create a categorical percentage bar plot.
            Args:
                column: Categorical column name.
                top_n: Maximum number of categories to display.
                include_missing: Include nulls as a category.
                rotate: X-axis label rotation in degrees.
            """
            return visualizer.percentage_bar_plot(
                column=column, top_n=top_n, include_missing=include_missing, rotate=rotate
            )

        @tool
        def categorical_horizontal_count_plot(
            column: str,
            top_n: int = 15,
            include_missing: bool = False,
        ) -> Any:
            """
            Create a horizontal categorical count plot.
            Args:
                column: Categorical column name.
                top_n: Maximum number of categories to display.
                include_missing: Include nulls as a category.
            """
            return visualizer.horizontal_count_plot(
                column=column, top_n=top_n, include_missing=include_missing
            )

        @tool
        def categorical_pareto_chart(
            column: str,
            top_n: int = 15,
            include_missing: bool = False,
        ) -> Any:
            """
            Create a Pareto chart for a categorical column.
            Args:
                column: Categorical column name.
                top_n: Maximum number of categories to display.
                include_missing: Include nulls in the distribution.
            """
            return visualizer.pareto_chart(
                column=column, top_n=top_n, include_missing=include_missing
            )

        @tool
        def categorical_pie_chart(
            column: str,
            top_n: int = 6,
            include_missing: bool = False,
        ) -> Any:
            """
            Create a pie chart of categorical proportions.
            Args:
                column: Categorical column name.
                top_n: Number of leading categories before Other.
                include_missing: Include nulls as a category.
            """
            return visualizer.pie_chart(
                column=column, top_n=top_n, include_missing=include_missing
            )

        @tool
        def categorical_missingness_plot(column: str) -> Any:
            """
            Create a present-versus-missing count plot.
            Args:
                column: Categorical column name.
            """
            return visualizer.missingness_plot(column=column)

        tools = [
            categorical_count_plot,
            categorical_percentage_plot,
            categorical_horizontal_count_plot,
            categorical_pareto_chart,
            categorical_pie_chart,
            categorical_missingness_plot,
        ]
       
        self.logger.info("Built %d categorical visualization tools.", len(tools))
        return tools

def build_categorical_univariate_visualization_tools(
    df: pd.DataFrame,
    top_n: int = 15,
) -> list[Any]:
    """
    Build all categorical univariate visualization tools.
    Example
    -------
    tools = build_categorical_univariate_visualization_tools(df)
    """
    return CategoricalUnivariateVisualization(df=df, top_n=top_n).build_tools()


# ====================================================
# Factory
# ====================================================

def build_categorical_univariate_statistics_tools(df: pd.DataFrame) -> list[Any]:
    """
    Create all categorical univariate LangChain tools for a DataFrame.
    Parameters
    ----------
    df:
        DataFrame that the tools will analyze.
    Returns
    -------
    list[Any]
        LangChain tools covering categorical univariate analysis.
    Raises
    ------
    CustomException
        If the DataFrame is invalid.
    """
    engine = CategoricalUnivariateStatistics(df)
    engine.logger.info("Building categorical univariate statistics tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d categorical univariate statistics tools.", len(tools))
    return tools

def build_categorical_univariate_visualization_tools(df: pd.DataFrame) -> list[Any]:
    """
    Build categorical univariate visualization tools.
    Parameters
    ----------
    df:
        Source DataFrame.
    Returns
    -------
    list[Any]
        LangChain visualization tools.
    Raises
    ------
    CustomException
        If the DataFrame is invalid.
    """
    engine = CategoricalUnivariateVisualization(df)
    engine.logger.info("Building categorical univariate visualization tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d categorical univariate visualization tools.", len(tools))
    return tools

# ====================================================
# Build Tools
# ====================================================
def build_categorical_univariate_tools(df: pd.DataFrame) -> list[Any]:
    """
    Build all categorical univariate tools.
    Includes:
    - Categorical statistics
    - Categorical visualizations
    """
    tools: list[Any] = []
    tools.extend(build_categorical_univariate_statistics_tools(df))
    tools.extend(build_categorical_univariate_visualization_tools(df))
    return tools