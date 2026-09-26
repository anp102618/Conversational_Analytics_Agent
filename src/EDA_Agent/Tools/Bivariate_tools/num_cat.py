# ====================================================
# Numeric-Categorical Bivariate Statistics
# ====================================================

"""
Numeric-Categorical Bivariate Exploratory Data Analysis.
Deterministic statistical analysis between:
    - One numeric feature
    - One categorical feature
The categorical feature is treated as the grouping variable and
the numeric feature is analyzed within each category.
The module is designed for:
    - Pandas Data Agent
    - LangChain tool calling
    - Deterministic EDA
    - Structured downstream analysis
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class NumericCategoricalBivariateStatistics:
    """
    Deterministic numeric-categorical bivariate statistical analysis.
    Parameters
    ----------
    df:
        Source DataFrame.
    Notes
    -----
    The numeric column is analyzed separately for each category
    of the categorical column.
    Example
    -------
    Numeric column:
        salary
    Categorical column:
        department
    The class can produce:
        - Group-wise descriptive statistics
        - Category counts
        - Central tendency comparison
        - Dispersion comparison
        - Quantile comparison
        - Group-wise missing values
        - Group-wise outlier analysis
        - ANOVA
        - Kruskal-Wallis test
        - Between-group variation
        - Within-group variation
        - Category ranking
    """
    DEFAULT_TOP_N = 15
    DEFAULT_OUTLIER_MULTIPLIER = 1.5

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the numeric-categorical statistics engine.
        Parameters
        ----------
        df:
            Source DataFrame.
        Raises
        ------
        CustomException
            If the DataFrame is invalid.
        """
        self.logger = get_log(__name__)
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("df must be a pandas DataFrame.")
            if df.empty:
                raise ValueError("DataFrame cannot be empty.")
            self.df = df.copy()
            self.logger.info(
                "Initialized NumericCategoricalBivariateStatistics with shape=%s",
                self.df.shape,
            )
        except Exception as exc:
            self.logger.exception(
                "Failed to initialize numeric-categorical statistics engine."
            )
            raise CustomException(exc) from exc

    def _validate_columns(
        self, numeric_column: str, categorical_column: str
    ) -> None:
        """
        Validate numeric and categorical column names and dtypes.
        """
        if not isinstance(numeric_column, str) or not numeric_column.strip():
            raise ValueError("numeric_column must be a non-empty string.")
        if not isinstance(categorical_column, str) or not categorical_column.strip():
            raise ValueError("categorical_column must be a non-empty string.")
        if numeric_column not in self.df.columns:
            raise KeyError(f"Numeric column '{numeric_column}' not found in DataFrame.")
        if categorical_column not in self.df.columns:
            raise KeyError(
                f"Categorical column '{categorical_column}' not found in DataFrame."
            )
        if numeric_column == categorical_column:
            raise ValueError("numeric_column and categorical_column must be different.")
        if not pd.api.types.is_numeric_dtype(self.df[numeric_column]):
            raise TypeError(f"Column '{numeric_column}' must be numeric.")
        if pd.api.types.is_numeric_dtype(self.df[categorical_column]):
            raise TypeError(f"Column '{categorical_column}' must be categorical.")

    def _get_pair(
        self, numeric_column: str, categorical_column: str
    ) -> pd.DataFrame:
        """
        Return a clean numeric-categorical pair.
        Missing values in either column are removed because
        group-wise statistics require both values to be present.
        """
        self._validate_columns(numeric_column, categorical_column)
        pair = self.df[[numeric_column, categorical_column]].dropna(
            subset=[numeric_column, categorical_column]
        )
        if pair.empty:
            raise ValueError(
                "No valid observations remain after removing missing values."
            )
        return pair

    @track_performance
    def group_statistics(
        self, numeric_column: str, categorical_column: str
    ) -> dict[str, Any]:
        """
        Calculate descriptive statistics of the numeric variable
        for every category.
        Returns
        -------
        dict[str, Any]
            Group-wise count, mean, median, std, min, max,
            quartiles and range.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column)
            grouped = pair.groupby(categorical_column, observed=True)[numeric_column]
            result = grouped.agg(
                count="count", mean="mean", median="median",
                std="std", min="min", max="max",
            )
            result["q1"] = grouped.quantile(0.25)
            result["q3"] = grouped.quantile(0.75)
            result["iqr"] = result["q3"] - result["q1"]
            result["range"] = result["max"] - result["min"]
            result = result.reset_index()
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "groups": result.to_dict(orient="records"),
                "group_count": int(len(result)),
            }
        except Exception as exc:
            self.logger.exception(
                "Failed group-wise statistics for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def category_counts(
        self, numeric_column: str, categorical_column: str
    ) -> dict[str, Any]:
        """
        Calculate observation counts for every category.
        The numeric column is included to ensure that only
        observations with valid numeric values participate.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column)
            counts = (
                pair[categorical_column]
                .value_counts(dropna=False)
                .rename("count")
                .reset_index()
            )
            counts.columns = [categorical_column, "count"]
            counts["percentage"] = counts["count"] / counts["count"].sum() * 100.0
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "total_observations": int(len(pair)),
                "categories": counts.to_dict(orient="records"),
            }
        except Exception as exc:
            self.logger.exception(
                "Failed category count analysis for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def central_tendency_comparison(
        self, numeric_column: str, categorical_column: str
    ) -> dict[str, Any]:
        """
        Compare mean and median of the numeric variable
        across categories.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column)
            grouped = pair.groupby(categorical_column, observed=True)[numeric_column]
            result = grouped.agg(
                count="count", mean="mean", median="median"
            ).reset_index()
            result["mean_median_difference"] = result["mean"] - result["median"]
            result["mean_median_ratio"] = np.where(
                result["median"] != 0, result["mean"] / result["median"], np.nan
            )
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "groups": result.to_dict(orient="records"),
            }
        except Exception as exc:
            self.logger.exception("Failed central tendency comparison.")
            raise CustomException(exc) from exc

    @track_performance
    def dispersion_comparison(
        self, numeric_column: str, categorical_column: str
    ) -> dict[str, Any]:
        """
        Compare variability of the numeric variable
        across categories.
        Includes:
            - Standard deviation
            - Variance
            - IQR
            - Range
            - Coefficient of variation
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column)
            grouped = pair.groupby(categorical_column, observed=True)[numeric_column]
            result = grouped.agg(
                count="count", mean="mean", std="std",
                variance="var", min="min", max="max",
            )
            result["iqr"] = grouped.quantile(0.75) - grouped.quantile(0.25)
            result["range"] = result["max"] - result["min"]
            result["coefficient_of_variation"] = np.where(
                result["mean"] != 0, result["std"] / result["mean"], np.nan
            )
            result = result.reset_index()
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "groups": result.to_dict(orient="records"),
            }
        except Exception as exc:
            self.logger.exception("Failed dispersion comparison.")
            raise CustomException(exc) from exc

    @track_performance
    def quantile_comparison(
        self, numeric_column: str, categorical_column: str
    ) -> dict[str, Any]:
        """
        Compare important quantiles of the numeric variable
        across categories.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column)
            grouped = pair.groupby(categorical_column, observed=True)[numeric_column]
            result = grouped.quantile(
                [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            ).unstack()
            result.columns = ["q01", "q05", "q25", "q50", "q75", "q95", "q99"]
            result = result.reset_index()
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "groups": result.to_dict(orient="records"),
            }
        except Exception as exc:
            self.logger.exception("Failed quantile comparison.")
            raise CustomException(exc) from exc

    @track_performance
    def missing_values_by_category(
        self, numeric_column: str, categorical_column: str
    ) -> dict[str, Any]:
        """
        Analyze missing numeric values within each category.
        Unlike most methods, this method intentionally does not
        remove missing numeric values before calculating the
        missingness statistics.
        """
        try:
            self._validate_columns(numeric_column, categorical_column)
            data = self.df[[numeric_column, categorical_column]].copy()
            result = (
                data.groupby(categorical_column, observed=True)[numeric_column]
                .agg(total="size", missing="count")
            )
            result["missing"] = result["total"] - result["missing"]
            result["missing_percentage"] = np.where(
                result["total"] > 0,
                result["missing"] / result["total"] * 100.0,
                0.0,
            )
            result = result.reset_index()
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "groups": result.to_dict(orient="records"),
            }
        except Exception as exc:
            self.logger.exception("Failed missing-value analysis by category.")
            raise CustomException(exc) from exc

    @track_performance
    def outliers_by_category(
        self,
        numeric_column: str,
        categorical_column: str,
        multiplier: float = DEFAULT_OUTLIER_MULTIPLIER,
    ) -> dict[str, Any]:
        """
        Detect numeric outliers separately within each category
        using the IQR method.
        Parameters
        ----------
        multiplier:
            IQR multiplier used for lower and upper bounds.
            Default is 1.5.
        """
        try:
            if multiplier <= 0:
                raise ValueError("multiplier must be greater than zero.")
            pair = self._get_pair(numeric_column, categorical_column)
            records: list[dict[str, Any]] = []
            for category, group in pair.groupby(categorical_column, observed=True):
                values = group[numeric_column]
                q1 = float(values.quantile(0.25))
                q3 = float(values.quantile(0.75))
                iqr = q3 - q1
                lower_bound = q1 - multiplier * iqr
                upper_bound = q3 + multiplier * iqr
                outlier_mask = (values < lower_bound) | (values > upper_bound)
                count = int(len(values))
                outlier_count = int(outlier_mask.sum())
                records.append({
                    categorical_column: category,
                    "count": count,
                    "q1": q1,
                    "q3": q3,
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_count": outlier_count,
                    "outlier_percentage": (
                        outlier_count / count * 100.0 if count > 0 else 0.0
                    ),
                })
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "iqr_multiplier": multiplier,
                "groups": records,
            }
        except Exception as exc:
            self.logger.exception("Failed outlier analysis by category.")
            raise CustomException(exc) from exc

    @track_performance
    def anova_test(
        self, numeric_column: str, categorical_column: str
    ) -> dict[str, Any]:
        """
        Perform one-way ANOVA to test whether numeric means
        differ across categories.
        H0:
            All category means are equal.
        H1:
            At least one category mean differs.
        Notes
        -----
        ANOVA assumes independent observations, approximately
        normal residuals, and reasonably homogeneous variances.
        The method reports the test result but does not interpret
        causality.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column)
            groups = [
                group[numeric_column].to_numpy()
                for _, group in pair.groupby(categorical_column, observed=True)
                if len(group) >= 2
            ]
            if len(groups) < 2:
                raise ValueError(
                    "ANOVA requires at least two categories "
                    "with two or more observations each."
                )
            statistic, p_value = f_oneway(*groups)
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "test": "one_way_anova",
                "groups_tested": len(groups),
                "f_statistic": float(statistic),
                "p_value": float(p_value),
                "significance_level": 0.05,
                "significant_at_0_05": bool(p_value < 0.05),
            }
        except Exception as exc:
            self.logger.exception("Failed one-way ANOVA test.")
            raise CustomException(exc) from exc

    @track_performance
    def kruskal_wallis_test(
        self, numeric_column: str, categorical_column: str
    ) -> dict[str, Any]:
        """
        Perform the Kruskal-Wallis H test across categories.
        This is a non-parametric alternative for comparing
        distributions across multiple independent groups.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column)
            groups = [
                group[numeric_column].to_numpy()
                for _, group in pair.groupby(categorical_column, observed=True)
                if len(group) >= 2
            ]
            if len(groups) < 2:
                raise ValueError(
                    "Kruskal-Wallis requires at least two categories "
                    "with two or more observations each."
                )
            statistic, p_value = kruskal(*groups)
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "test": "kruskal_wallis",
                "groups_tested": len(groups),
                "h_statistic": float(statistic),
                "p_value": float(p_value),
                "significance_level": 0.05,
                "significant_at_0_05": bool(p_value < 0.05),
            }
        except Exception as exc:
            self.logger.exception("Failed Kruskal-Wallis test.")
            raise CustomException(exc) from exc

    @track_performance
    def category_ranking(
        self,
        numeric_column: str,
        categorical_column: str,
        statistic: str = "mean",
        ascending: bool = False,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:
        """
        Rank categories according to a numeric summary statistic.
        Parameters
        ----------
        statistic:
            One of:
                - mean
                - median
                - std
                - min
                - max
        ascending:
            Sort ascending when True.
        top_n:
            Maximum number of categories returned.
        """
        try:
            allowed_statistics = {"mean", "median", "std", "min", "max"}
            if statistic not in allowed_statistics:
                raise ValueError(
                    f"statistic must be one of {sorted(allowed_statistics)}."
                )
            if top_n <= 0:
                raise ValueError("top_n must be greater than zero.")
            pair = self._get_pair(numeric_column, categorical_column)
            grouped = (
                pair.groupby(categorical_column, observed=True)[numeric_column]
                .agg(
                    count="count", mean="mean", median="median",
                    std="std", min="min", max="max",
                )
                .reset_index()
            )
            grouped = grouped.sort_values(by=statistic, ascending=ascending).head(top_n)
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "ranking_statistic": statistic,
                "ascending": ascending,
                "top_n": top_n,
                "categories": grouped.to_dict(orient="records"),
            }
        except Exception as exc:
            self.logger.exception("Failed category ranking.")
            raise CustomException(exc) from exc

    @track_performance
    def variance_decomposition(
        self, numeric_column: str, categorical_column: str
    ) -> dict[str, Any]:
        """
        Decompose total numeric variation into:
            - Between-group sum of squares
            - Within-group sum of squares
            - Total sum of squares
        This provides deterministic evidence about how much
        variation is associated with differences between
        category means.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column)
            values = pair[numeric_column]
            overall_mean = float(values.mean())
            grouped = pair.groupby(categorical_column, observed=True)[numeric_column]
            group_stats = grouped.agg(count="count", mean="mean")
            between_ss = float(
                (group_stats["count"] * (group_stats["mean"] - overall_mean) ** 2).sum()
            )
            within_ss = 0.0
            for _, group in grouped:
                within_ss += float(((group - group.mean()) ** 2).sum())
            total_ss = float(((values - overall_mean) ** 2).sum())
            between_ratio = between_ss / total_ss if total_ss > 0 else np.nan
            within_ratio = within_ss / total_ss if total_ss > 0 else np.nan
            return {
                "numeric_column": numeric_column,
                "categorical_column": categorical_column,
                "overall_mean": overall_mean,
                "between_group_sum_of_squares": between_ss,
                "within_group_sum_of_squares": within_ss,
                "total_sum_of_squares": total_ss,
                "between_group_variance_ratio": (
                    float(between_ratio) if not np.isnan(between_ratio) else None
                ),
                "within_group_variance_ratio": (
                    float(within_ratio) if not np.isnan(within_ratio) else None
                ),
            }
        except Exception as exc:
            self.logger.exception("Failed variance decomposition.")
            raise CustomException(exc) from exc

    @track_performance
    def build_tools(self) -> list[Any]:
        """
        Build all numeric-categorical bivariate statistics
        as LangChain tools.
        Returns
        -------
        list[Any]
            LangChain tools.
        """
        engine = self

        @tool
        def group_statistics(
            numeric_column: str, categorical_column: str
        ) -> dict[str, Any]:
            """
            Calculate descriptive statistics of a numeric column
            separately for each category.
            Use this for questions about mean, median, standard
            deviation, min, max, quartiles, IQR, or range by category.
            """
            return engine.group_statistics(numeric_column, categorical_column)

        @tool
        def category_counts(
            numeric_column: str, categorical_column: str
        ) -> dict[str, Any]:
            """
            Calculate observation counts and percentages for
            each category.
            """
            return engine.category_counts(numeric_column, categorical_column)

        @tool
        def central_tendency_comparison(
            numeric_column: str, categorical_column: str
        ) -> dict[str, Any]:
            """
            Compare mean and median of a numeric column across
            categories.
            """
            return engine.central_tendency_comparison(
                numeric_column, categorical_column
            )

        @tool
        def dispersion_comparison(
            numeric_column: str, categorical_column: str
        ) -> dict[str, Any]:
            """
            Compare standard deviation, variance, IQR, range,
            and coefficient of variation across categories.
            """
            return engine.dispersion_comparison(numeric_column, categorical_column)

        @tool
        def quantile_comparison(
            numeric_column: str, categorical_column: str
        ) -> dict[str, Any]:
            """
            Compare important numeric quantiles across categories.
            """
            return engine.quantile_comparison(numeric_column, categorical_column)

        @tool
        def missing_values_by_category(
            numeric_column: str, categorical_column: str
        ) -> dict[str, Any]:
            """
            Analyze missing numeric values within each category.
            """
            return engine.missing_values_by_category(
                numeric_column, categorical_column
            )

        @tool
        def outliers_by_category(
            numeric_column: str,
            categorical_column: str,
            multiplier: float = 1.5,
        ) -> dict[str, Any]:
            """
            Detect IQR-based outliers separately within each category.
            """
            return engine.outliers_by_category(
                numeric_column, categorical_column, multiplier
            )

        @tool
        def anova_test(
            numeric_column: str, categorical_column: str
        ) -> dict[str, Any]:
            """
            Perform one-way ANOVA to test whether numeric means
            differ across categories.
            """
            return engine.anova_test(numeric_column, categorical_column)

        @tool
        def kruskal_wallis_test(
            numeric_column: str, categorical_column: str
        ) -> dict[str, Any]:
            """
            Perform the Kruskal-Wallis non-parametric test across
            categories.
            """
            return engine.kruskal_wallis_test(numeric_column, categorical_column)

        @tool
        def category_ranking(
            numeric_column: str,
            categorical_column: str,
            statistic: str = "mean",
            ascending: bool = False,
            top_n: int = 15,
        ) -> dict[str, Any]:
            """
            Rank categories according to a numeric statistic such
            as mean, median, standard deviation, minimum, or maximum.
            """
            return engine.category_ranking(
                numeric_column, categorical_column, statistic, ascending, top_n
            )

        @tool
        def variance_decomposition(
            numeric_column: str, categorical_column: str
        ) -> dict[str, Any]:
            """
            Decompose total numeric variation into between-category
            and within-category variation.
            """
            return engine.variance_decomposition(numeric_column, categorical_column)

        tools: list[Any] = [
            group_statistics,
            category_counts, 
            central_tendency_comparison,
            dispersion_comparison, 
            quantile_comparison, 
            missing_values_by_category,
            outliers_by_category, 
            anova_test, 
            kruskal_wallis_test,
            category_ranking, 
            variance_decomposition,
        ]
        self.logger.info(
            "Built %d numeric-categorical bivariate statistics tools.", len(tools)
        )
        return tools

#=================================================================================

# ====================================================
# Numeric-Categorical Bivariate Visualization
# ====================================================

"""
Numeric-Categorical Bivariate Visualization.
Deterministic visualization between:
    - One numeric feature
    - One categorical feature
The categorical feature is used as the grouping variable.
Uses:
    - Pandas
    - Seaborn
    - Matplotlib
    - LangChain tools
"""
from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class NumericCategoricalBivariateVisualization:
    """
    Numeric-categorical bivariate visualization engine.
    Parameters
    ----------
    df:
        Source DataFrame.
    figsize:
        Default figure size.
    Notes
    -----
    The numeric column is plotted against categories of the
    categorical column.
    """
    DEFAULT_TOP_N = 15
    DEFAULT_FIGSIZE = (12, 6)

    def __init__(
        self,
        df: pd.DataFrame,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    ) -> None:
        """
        Initialize the visualization engine.
        """
        self.logger = get_log(__name__)
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("df must be a pandas DataFrame.")
            if df.empty:
                raise ValueError("DataFrame cannot be empty.")
            if not isinstance(figsize, tuple) or len(figsize) != 2:
                raise ValueError("figsize must be a tuple of (width, height).")
            self.df = df.copy()
            self.figsize = figsize
            self.logger.info(
                "Initialized NumericCategoricalBivariateVisualization with shape=%s",
                self.df.shape,
            )
        except Exception as exc:
            self.logger.exception(
                "Failed to initialize numeric-categorical visualization engine."
            )
            raise CustomException(exc) from exc

    def _validate_columns(
        self, numeric_column: str, categorical_column: str
    ) -> None:
        """
        Validate numeric and categorical columns.
        """
        if not isinstance(numeric_column, str) or not numeric_column.strip():
            raise ValueError("numeric_column must be a non-empty string.")
        if not isinstance(categorical_column, str) or not categorical_column.strip():
            raise ValueError("categorical_column must be a non-empty string.")
        if numeric_column not in self.df.columns:
            raise KeyError(
                f"Numeric column '{numeric_column}' not found in DataFrame."
            )
        if categorical_column not in self.df.columns:
            raise KeyError(
                f"Categorical column '{categorical_column}' not found in DataFrame."
            )
        if numeric_column == categorical_column:
            raise ValueError(
                "numeric_column and categorical_column must be different."
            )
        if not pd.api.types.is_numeric_dtype(self.df[numeric_column]):
            raise TypeError(f"Column '{numeric_column}' must be numeric.")
        if pd.api.types.is_numeric_dtype(self.df[categorical_column]):
            raise TypeError(f"Column '{categorical_column}' must be categorical.")

    def _get_pair(
        self,
        numeric_column: str,
        categorical_column: str,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """
        Prepare numeric-categorical data for visualization.
        Missing observations are removed.
        When top_n is provided, only the most frequent
        categories are retained.
        """
        self._validate_columns(numeric_column, categorical_column)
        pair = self.df[[numeric_column, categorical_column]].dropna(
            subset=[numeric_column, categorical_column]
        )
        if pair.empty:
            raise ValueError(
                "No valid observations remain after removing missing values."
            )
        if top_n is not None:
            if top_n <= 0:
                raise ValueError("top_n must be greater than zero.")
            top_categories = (
                pair[categorical_column].value_counts().head(top_n).index
            )
            pair = pair[pair[categorical_column].isin(top_categories)]
        if pair.empty:
            raise ValueError("No observations remain after category filtering.")
        return pair

    def _create_figure(self, title: str) -> tuple[plt.Figure, plt.Axes]:
        """
        Create a Matplotlib figure and axes.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title)
        return fig, ax

    def _format_axis(
        self, ax: plt.Axes, numeric_column: str, categorical_column: str
    ) -> None:
        """
        Apply common axis formatting.
        """
        ax.set_xlabel(categorical_column)
        ax.set_ylabel(numeric_column)
        ax.tick_params(axis="x", rotation=45)
        ax.get_figure().tight_layout()

    @track_performance
    def box_plot(
        self,
        numeric_column: str,
        categorical_column: str,
        top_n: int = DEFAULT_TOP_N,
        show_points: bool = False,
    ) -> plt.Figure:
        """
        Create a box plot of the numeric variable by category.
        Shows:
            - Median
            - Quartiles
            - IQR
            - Potential outliers
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column, top_n)
            fig, ax = self._create_figure(
                f"{numeric_column} by {categorical_column}"
            )
            sns.boxplot(
                data=pair, x=categorical_column, y=numeric_column, ax=ax
            )
            if show_points:
                sns.stripplot(
                    data=pair, x=categorical_column, y=numeric_column,
                    ax=ax, alpha=0.35, size=3,
                )
            self._format_axis(ax, numeric_column, categorical_column)
            return fig
        except Exception as exc:
            self.logger.exception(
                "Failed box plot for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def violin_plot(
        self,
        numeric_column: str,
        categorical_column: str,
        top_n: int = DEFAULT_TOP_N,
        show_box: bool = True,
    ) -> plt.Figure:
        """
        Create a violin plot of the numeric distribution
        across categories.
        Shows:
            - Distribution shape
            - Density
            - Median
            - Quartiles when show_box=True
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column, top_n)
            fig, ax = self._create_figure(
                f"Distribution of {numeric_column} by {categorical_column}"
            )
            sns.violinplot(
                data=pair, x=categorical_column, y=numeric_column,
                inner="box" if show_box else None, ax=ax,
            )
            self._format_axis(ax, numeric_column, categorical_column)
            return fig
        except Exception as exc:
            self.logger.exception(
                "Failed violin plot for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def strip_plot(
        self,
        numeric_column: str,
        categorical_column: str,
        top_n: int = DEFAULT_TOP_N,
        jitter: float = 0.25,
    ) -> plt.Figure:
        """
        Create a strip plot showing individual observations
        for each category.
        """
        try:
            if jitter < 0:
                raise ValueError("jitter must be non-negative.")
            pair = self._get_pair(numeric_column, categorical_column, top_n)
            fig, ax = self._create_figure(
                f"{numeric_column} Observations by {categorical_column}"
            )
            sns.stripplot(
                data=pair, x=categorical_column, y=numeric_column,
                jitter=jitter, alpha=0.6, ax=ax,
            )
            self._format_axis(ax, numeric_column, categorical_column)
            return fig
        except Exception as exc:
            self.logger.exception(
                "Failed strip plot for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def swarm_plot(
        self,
        numeric_column: str,
        categorical_column: str,
        top_n: int = DEFAULT_TOP_N,
    ) -> plt.Figure:
        """
        Create a swarm plot showing individual observations
        while reducing point overlap.
        Best suited to small or moderately sized datasets.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column, top_n)
            fig, ax = self._create_figure(
                f"{numeric_column} Swarm Plot by {categorical_column}"
            )
            sns.swarmplot(
                data=pair, x=categorical_column, y=numeric_column, ax=ax
            )
            self._format_axis(ax, numeric_column, categorical_column)
            return fig
        except Exception as exc:
            self.logger.exception(
                "Failed swarm plot for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def mean_bar_plot(
        self,
        numeric_column: str,
        categorical_column: str,
        top_n: int = DEFAULT_TOP_N,
        errorbar: str | None = "sd",
    ) -> plt.Figure:
        """
        Plot the mean numeric value for each category.
        Parameters
        ----------
        errorbar:
            Error representation supported by Seaborn.
            Examples:
                "sd"
                "se"
                "pi"
                None
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column, top_n)
            fig, ax = self._create_figure(
                f"Mean {numeric_column} by {categorical_column}"
            )
            sns.barplot(
                data=pair, x=categorical_column, y=numeric_column,
                errorbar=errorbar, ax=ax,
            )
            self._format_axis(ax, numeric_column, categorical_column)
            return fig
        except Exception as exc:
            self.logger.exception(
                "Failed mean bar plot for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def median_bar_plot(
        self,
        numeric_column: str,
        categorical_column: str,
        top_n: int = DEFAULT_TOP_N,
    ) -> plt.Figure:
        """
        Plot the median numeric value for each category.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column, top_n)
            median_values = (
                pair.groupby(categorical_column, observed=True)[numeric_column]
                .median()
                .sort_values(ascending=False)
            )
            fig, ax = self._create_figure(
                f"Median {numeric_column} by {categorical_column}"
            )
            sns.barplot(x=median_values.index, y=median_values.values, ax=ax)
            self._format_axis(ax, numeric_column, categorical_column)
            return fig
        except Exception as exc:
            self.logger.exception(
                "Failed median bar plot for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def point_plot(
        self,
        numeric_column: str,
        categorical_column: str,
        top_n: int = DEFAULT_TOP_N,
        errorbar: str | None = "ci",
    ) -> plt.Figure:
        """
        Create a point plot showing the category-wise numeric
        central tendency and uncertainty.
        """
        try:
            pair = self._get_pair(numeric_column, categorical_column, top_n)
            fig, ax = self._create_figure(
                f"{numeric_column} Point Plot by {categorical_column}"
            )
            sns.pointplot(
                data=pair, x=categorical_column, y=numeric_column,
                errorbar=errorbar, ax=ax,
            )
            self._format_axis(ax, numeric_column, categorical_column)
            return fig
        except Exception as exc:
            self.logger.exception(
                "Failed point plot for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def distribution_plot(
        self,
        numeric_column: str,
        categorical_column: str,
        top_n: int = DEFAULT_TOP_N,
        bins: int = 30,
        kde: bool = False,
        multiple: str = "layer",
    ) -> plt.Figure:
        """
        Plot the numeric distribution across categories.
        Parameters
        ----------
        bins:
            Number of histogram bins.
        kde:
            Whether to overlay KDE curves.
        multiple:
            Seaborn histogram layout.
            Examples:
                "layer"
                "stack"
                "dodge"
                "fill"
        """
        try:
            if bins <= 0:
                raise ValueError("bins must be greater than zero.")
            allowed_multiple = {"layer", "stack", "dodge", "fill"}
            if multiple not in allowed_multiple:
                raise ValueError(
                    f"multiple must be one of {sorted(allowed_multiple)}."
                )
            pair = self._get_pair(numeric_column, categorical_column, top_n)
            fig, ax = self._create_figure(
                f"{numeric_column} Distribution by {categorical_column}"
            )
            sns.histplot(
                data=pair, x=numeric_column, hue=categorical_column,
                bins=bins, kde=kde, multiple=multiple, ax=ax,
            )
            ax.set_xlabel(numeric_column)
            ax.set_ylabel("Count")
            fig.tight_layout()
            return fig
        except Exception as exc:
            self.logger.exception(
                "Failed distribution plot for %s by %s.",
                numeric_column, categorical_column,
            )
            raise CustomException(exc) from exc

    @track_performance
    def build_tools(self) -> list[Any]:
        """
        Build all numeric-categorical visualization tools
        as LangChain tools.
        Returns
        -------
        list[Any]
            LangChain visualization tools.
        """
        engine = self

        @tool
        def box_plot(
            numeric_column: str,
            categorical_column: str,
            top_n: int = 15,
            show_points: bool = False,
        ) -> plt.Figure:
            """
            Create a box plot of a numeric variable across categories.
            Useful for comparing median, quartiles, spread, and outliers.
            """
            return engine.box_plot(
                numeric_column, categorical_column, top_n, show_points
            )

        @tool
        def violin_plot(
            numeric_column: str,
            categorical_column: str,
            top_n: int = 15,
            show_box: bool = True,
        ) -> plt.Figure:
            """
            Create a violin plot to compare numeric distributions
            across categories.
            """
            return engine.violin_plot(
                numeric_column, categorical_column, top_n, show_box
            )

        @tool
        def strip_plot(
            numeric_column: str,
            categorical_column: str,
            top_n: int = 15,
            jitter: float = 0.25,
        ) -> plt.Figure:
            """
            Show individual numeric observations across categories.
            """
            return engine.strip_plot(
                numeric_column, categorical_column, top_n, jitter
            )

        @tool
        def swarm_plot(
            numeric_column: str,
            categorical_column: str,
            top_n: int = 15,
        ) -> plt.Figure:
            """
            Create a swarm plot showing individual observations
            across categories.
            """
            return engine.swarm_plot(numeric_column, categorical_column, top_n)

        @tool
        def mean_bar_plot(
            numeric_column: str,
            categorical_column: str,
            top_n: int = 15,
            errorbar: str | None = "sd",
        ) -> plt.Figure:
            """
            Compare category-wise mean values of a numeric variable.
            """
            return engine.mean_bar_plot(
                numeric_column, categorical_column, top_n, errorbar
            )

        @tool
        def median_bar_plot(
            numeric_column: str,
            categorical_column: str,
            top_n: int = 15,
        ) -> plt.Figure:
            """
            Compare category-wise median values of a numeric variable.
            """
            return engine.median_bar_plot(
                numeric_column, categorical_column, top_n
            )

        @tool
        def point_plot(
            numeric_column: str,
            categorical_column: str,
            top_n: int = 15,
            errorbar: str | None = "ci",
        ) -> plt.Figure:
            """
            Compare category-wise numeric central tendency with
            uncertainty intervals.
            """
            return engine.point_plot(
                numeric_column, categorical_column, top_n, errorbar
            )

        @tool
        def distribution_plot(
            numeric_column: str,
            categorical_column: str,
            top_n: int = 15,
            bins: int = 30,
            kde: bool = False,
            multiple: str = "layer",
        ) -> plt.Figure:
            """
            Compare numeric distributions across categories using
            histograms, optionally with KDE.
            """
            return engine.distribution_plot(
                numeric_column, categorical_column, top_n, bins, kde, multiple
            )

        tools: list[Any] = [
            box_plot,
            violin_plot, 
            strip_plot, 
            swarm_plot,
            mean_bar_plot, 
            median_bar_plot, 
            point_plot, 
            distribution_plot,
        ]
        self.logger.info(
            "Built %d numeric-categorical bivariate visualization tools.",
            len(tools),
        )
        return tools


# ====================================================
# Factory
# ====================================================
def build_numeric_categorical_bivariate_statistics_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Create all numeric-categorical bivariate statistical LangChain
    tools for a DataFrame.
    Parameters
    ----------
    df:
        DataFrame that the tools will analyze.
    Returns
    -------
    list[Any]
        LangChain tools covering numeric-categorical bivariate analysis.
    Raises
    ------
    CustomException
        If the DataFrame is invalid.
    """
    engine = NumericCategoricalBivariateStatistics(df)
    engine.logger.info("Building numeric-categorical bivariate statistics tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d numeric-categorical bivariate statistics tools.", len(tools))
    return tools

def build_numeric_categorical_bivariate_visualization_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Build numeric-categorical bivariate visualization tools.
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
    engine = NumericCategoricalBivariateVisualization(df)
    engine.logger.info("Building numeric-categorical bivariate visualization tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d numeric-categorical bivariate visualization tools.", len(tools))
    return tools

# ====================================================
# Build Tools
# ====================================================
def build_numeric_categorical_bivariate_tools(df: pd.DataFrame) -> list[Any]:
    """
    Build all numeric-categorical bivariate tools.
    Includes
    --------
    - Numeric-categorical bivariate statistics
    - Numeric-categorical bivariate visualizations
    Parameters
    ----------
    df:
        Source DataFrame.
    Returns
    -------
    list[Any]
        Complete numeric-categorical bivariate tool collection.
    """
    tools: list[Any] = []
    tools.extend(build_numeric_categorical_bivariate_statistics_tools(df))
    tools.extend(build_numeric_categorical_bivariate_visualization_tools(df))
    return tools

#=====================================================================

