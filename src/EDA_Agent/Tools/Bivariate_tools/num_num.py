
# ====================================================
# Numeric-Numeric Bivariate Statistics
# ====================================================
"""
Numeric-Numeric Bivariate Statistics
====================================
Deterministic bivariate statistical analysis tools for the
Pandas Data Agent.
Supported analyses
------------------
- Basic paired statistics
- Pearson correlation
- Spearman correlation
- Kendall correlation
- Covariance
- Linear regression
- R-squared
- Regression residual diagnostics
- Paired missing-value analysis
- Numeric relationship summary
Design principles
-----------------
1. Pandas performs deterministic calculations.
2. The LLM selects tools and supplies analytical parameters.
3. The DataFrame remains internal to the engine.
4. Every tool validates both requested columns.
5. Tool results use JSON-compatible structured dictionaries.
6. Project logging and CustomException are used consistently.
7. Source DataFrame is never modified.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy.stats import kendalltau, linregress, pearsonr, spearmanr
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class NumericNumericBivariateStatistics:
    """
    Deterministic numeric-numeric bivariate analysis engine.
    Parameters
    ----------
    df:
        Pandas DataFrame containing the dataset.
    logger_name:
        Logger name used by the application.
    Notes
    -----
    The engine does not modify the source DataFrame or write files.
    Both columns are treated as numeric variables and pairwise
    complete observations are used for relationship analysis.
    """
    DEFAULT_MIN_OBSERVATIONS = 3

    def __init__(
        self,
        df: pd.DataFrame,
        logger_name: str = "PandasAgentNumericNumericBI",
    ) -> None:
        """Initialize the numeric-numeric bivariate analysis engine."""
        if df is None:
            raise CustomException("DataFrame cannot be None.", get_log(logger_name))
        if not isinstance(df, pd.DataFrame):
            raise CustomException("df must be a pandas DataFrame.", get_log(logger_name))
        if df.empty:
            raise CustomException("DataFrame cannot be empty.", get_log(logger_name))
        self.df = df
        self.logger = get_log(logger_name)
        self.logger.info(
            "Initialized numeric-numeric bivariate statistics engine "
            "with %d rows and %d columns.",
            df.shape[0], df.shape[1],
        )

    def _validate_columns(
        self, x_column: str, y_column: str
    ) -> tuple[pd.Series, pd.Series]:
        """
        Validate and return two numeric columns.
        Raises
        ------
        ValueError
            If column names are invalid, columns do not exist,
            columns are identical, or columns are not numeric.
        """
        try:
            if not isinstance(x_column, str) or not x_column.strip():
                raise ValueError("x_column must be a non-empty string.")
            if not isinstance(y_column, str) or not y_column.strip():
                raise ValueError("y_column must be a non-empty string.")
            if x_column not in self.df.columns:
                raise ValueError(f"Column '{x_column}' does not exist.")
            if y_column not in self.df.columns:
                raise ValueError(f"Column '{y_column}' does not exist.")
            if x_column == y_column:
                raise ValueError("x_column and y_column must be different.")
            x_series, y_series = self.df[x_column], self.df[y_column]
            if not pd.api.types.is_numeric_dtype(x_series):
                raise ValueError(
                    f"Column '{x_column}' must be numeric. Detected dtype: {x_series.dtype}"
                )
            if not pd.api.types.is_numeric_dtype(y_series):
                raise ValueError(
                    f"Column '{y_column}' must be numeric. Detected dtype: {y_series.dtype}"
                )
            return x_series, y_series
        except Exception as exc:
            self.logger.error(
                "Numeric bivariate column validation failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    def _get_paired_data(self, x_column: str, y_column: str) -> pd.DataFrame:
        """
        Return pairwise complete observations.
        Rows where either X or Y is missing are removed only from
        the temporary analytical dataset.
        """
        x_series, y_series = self._validate_columns(x_column, y_column)
        return pd.DataFrame({"x": x_series, "y": y_series}).dropna()

    def _validate_min_observations(
        self, paired: pd.DataFrame, minimum: int = DEFAULT_MIN_OBSERVATIONS
    ) -> None:
        """Validate that enough paired observations exist."""
        if minimum < 2:
            raise ValueError("minimum observations must be at least 2.")
        if len(paired) < minimum:
            raise ValueError(
                f"At least {minimum} paired observations are required; "
                f"only {len(paired)} are available."
            )

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert Pandas and NumPy values into JSON-safe values."""
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

    def _result(
        self, tool_name: str, x_column: str, y_column: str, **values: Any
    ) -> dict[str, Any]:
        """Build standardized JSON-compatible result."""
        result = {
            "status": "success",
            "tool": tool_name,
            "analysis_type": "bivariate",
            "data_type": "numeric_numeric",
            "x_column": x_column,
            "y_column": y_column,
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

    @track_performance
    def paired_statistics(self, x_column: str, y_column: str) -> dict[str, Any]:
        """
        Calculate basic statistics for paired numeric variables.
        Includes:
        - Total rows
        - Valid X observations
        - Valid Y observations
        - Complete paired observations
        - Pairwise missing observations
        - X/Y descriptive statistics
        - X/Y means
        - X/Y standard deviations
        - X/Y ranges
        """
        try:
            x_series, y_series = self._validate_columns(x_column, y_column)
            paired = self._get_paired_data(x_column, y_column)
            total_rows, paired_count = len(self.df), len(paired)
            return self._result(
                "paired_statistics", x_column, y_column,
                total_rows=total_rows,
                x_valid_count=int(x_series.notna().sum()),
                y_valid_count=int(y_series.notna().sum()),
                paired_valid_count=paired_count,
                paired_missing_count=total_rows - paired_count,
                paired_missing_percentage=round(
                    (total_rows - paired_count) / total_rows * 100, 4
                ),
                x_mean=float(paired["x"].mean()) if paired_count else None,
                y_mean=float(paired["y"].mean()) if paired_count else None,
                x_median=float(paired["x"].median()) if paired_count else None,
                y_median=float(paired["y"].median()) if paired_count else None,
                x_std=float(paired["x"].std()) if paired_count > 1 else None,
                y_std=float(paired["y"].std()) if paired_count > 1 else None,
                x_min=float(paired["x"].min()) if paired_count else None,
                x_max=float(paired["x"].max()) if paired_count else None,
                y_min=float(paired["y"].min()) if paired_count else None,
                y_max=float(paired["y"].max()) if paired_count else None,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Failed paired statistics for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def pearson_correlation(self, x_column: str, y_column: str) -> dict[str, Any]:
        """
        Calculate Pearson correlation coefficient.
        Measures linear association between two numeric variables.
        Returns:
        - correlation coefficient
        - p-value
        - observation count
        """
        try:
            paired = self._get_paired_data(x_column, y_column)
            self._validate_min_observations(paired)
            if paired["x"].nunique() < 2:
                raise ValueError(f"Column '{x_column}' has zero variance.")
            if paired["y"].nunique() < 2:
                raise ValueError(f"Column '{y_column}' has zero variance.")
            statistic, p_value = pearsonr(paired["x"], paired["y"])
            return self._result(
                "pearson_correlation", x_column, y_column,
                observation_count=len(paired),
                correlation=float(statistic),
                p_value=float(p_value),
                absolute_correlation=abs(float(statistic)),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Pearson correlation failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def spearman_correlation(self, x_column: str, y_column: str) -> dict[str, Any]:
        """
        Calculate Spearman rank correlation.
        Measures monotonic association and is less dependent
        on a strictly linear relationship.
        """
        try:
            paired = self._get_paired_data(x_column, y_column)
            self._validate_min_observations(paired)
            if paired["x"].nunique() < 2:
                raise ValueError(f"Column '{x_column}' has insufficient variation.")
            if paired["y"].nunique() < 2:
                raise ValueError(f"Column '{y_column}' has insufficient variation.")
            statistic, p_value = spearmanr(paired["x"], paired["y"])
            return self._result(
                "spearman_correlation", x_column, y_column,
                observation_count=len(paired),
                correlation=float(statistic),
                p_value=float(p_value),
                absolute_correlation=abs(float(statistic)),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Spearman correlation failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def kendall_correlation(self, x_column: str, y_column: str) -> dict[str, Any]:
        """
        Calculate Kendall's tau correlation.
        Useful for measuring ordinal/monotonic association
        based on concordant and discordant pairs.
        """
        try:
            paired = self._get_paired_data(x_column, y_column)
            self._validate_min_observations(paired)
            if paired["x"].nunique() < 2:
                raise ValueError(f"Column '{x_column}' has insufficient variation.")
            if paired["y"].nunique() < 2:
                raise ValueError(f"Column '{y_column}' has insufficient variation.")
            statistic, p_value = kendalltau(paired["x"], paired["y"])
            return self._result(
                "kendall_correlation", x_column, y_column,
                observation_count=len(paired),
                correlation=float(statistic),
                p_value=float(p_value),
                absolute_correlation=abs(float(statistic)),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Kendall correlation failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def covariance(self, x_column: str, y_column: str) -> dict[str, Any]:
        """Calculate sample covariance between two numeric variables."""
        try:
            paired = self._get_paired_data(x_column, y_column)
            self._validate_min_observations(paired)
            return self._result(
                "covariance", x_column, y_column,
                observation_count=len(paired),
                covariance=float(paired["x"].cov(paired["y"])),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Covariance calculation failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def linear_regression(self, x_column: str, y_column: str) -> dict[str, Any]:
        """
        Fit a simple ordinary least-squares linear regression.
        Model:
            y = intercept + slope * x
        Returns:
        - slope
        - intercept
        - r-value
        - R-squared
        - p-value
        - standard error
        """
        try:
            paired = self._get_paired_data(x_column, y_column)
            self._validate_min_observations(paired)
            if paired["x"].nunique() < 2:
                raise ValueError(f"Column '{x_column}' has zero variance.")
            regression = linregress(paired["x"], paired["y"])
            return self._result(
                "linear_regression", x_column, y_column,
                observation_count=len(paired),
                slope=float(regression.slope),
                intercept=float(regression.intercept),
                r_value=float(regression.rvalue),
                r_squared=float(regression.rvalue ** 2),
                p_value=float(regression.pvalue),
                standard_error=float(regression.stderr),
                intercept_standard_error=(
                    float(regression.intercept_stderr)
                    if hasattr(regression, "intercept_stderr") else None
                ),
                equation=(
                    f"{y_column} = {regression.intercept:.6f} + "
                    f"{regression.slope:.6f} * {x_column}"
                ),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Linear regression failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def regression_residuals(self, x_column: str, y_column: str) -> dict[str, Any]:
        """
        Calculate residual diagnostics for simple linear regression.
        Returns:
        - residual mean
        - residual standard deviation
        - MAE
        - MSE
        - RMSE
        - minimum residual
        - maximum residual
        """
        try:
            paired = self._get_paired_data(x_column, y_column)
            self._validate_min_observations(paired)
            if paired["x"].nunique() < 2:
                raise ValueError(f"Column '{x_column}' has zero variance.")
            regression = linregress(paired["x"], paired["y"])
            predictions = regression.intercept + regression.slope * paired["x"]
            residuals = paired["y"] - predictions
            mae = float(np.mean(np.abs(residuals)))
            mse = float(np.mean(residuals ** 2))
            rmse = float(np.sqrt(mse))
            return self._result(
                "regression_residuals", x_column, y_column,
                observation_count=len(paired),
                residual_mean=float(residuals.mean()),
                residual_std=float(residuals.std(ddof=1)),
                mean_absolute_error=mae,
                mean_squared_error=mse,
                root_mean_squared_error=rmse,
                residual_min=float(residuals.min()),
                residual_max=float(residuals.max()),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Regression residual analysis failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def paired_missing_values(self, x_column: str, y_column: str) -> dict[str, Any]:
        """
        Analyze missingness across a numeric column pair.
        Separates:
        - X only missing
        - Y only missing
        - both missing
        - complete pairs
        """
        try:
            x_series, y_series = self._validate_columns(x_column, y_column)
            x_missing, y_missing = x_series.isna(), y_series.isna()
            both_missing = x_missing & y_missing
            x_only_missing = x_missing & ~y_missing
            y_only_missing = ~x_missing & y_missing
            complete = ~x_missing & ~y_missing
            total = len(self.df)
            return self._result(
                "paired_missing_values", x_column, y_column,
                total_rows=total,
                x_missing_count=int(x_missing.sum()),
                y_missing_count=int(y_missing.sum()),
                both_missing_count=int(both_missing.sum()),
                x_only_missing_count=int(x_only_missing.sum()),
                y_only_missing_count=int(y_only_missing.sum()),
                complete_pair_count=int(complete.sum()),
                complete_pair_percentage=round(complete.sum() / total * 100, 4),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Paired missing-value analysis failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def relationship_summary(self, x_column: str, y_column: str) -> dict[str, Any]:
        """
        Produce a compact deterministic summary of the relationship.
        Includes:
        - paired observation count
        - Pearson correlation
        - Spearman correlation
        - Kendall correlation
        - covariance
        - linear regression
        - R-squared
        """
        try:
            paired = self._get_paired_data(x_column, y_column)
            self._validate_min_observations(paired)
            if paired["x"].nunique() < 2:
                raise ValueError(f"Column '{x_column}' has zero variance.")
            if paired["y"].nunique() < 2:
                raise ValueError(f"Column '{y_column}' has zero variance.")
            pearson_value, pearson_p = pearsonr(paired["x"], paired["y"])
            spearman_value, spearman_p = spearmanr(paired["x"], paired["y"])
            kendall_value, kendall_p = kendalltau(paired["x"], paired["y"])
            regression = linregress(paired["x"], paired["y"])
            return self._result(
                "relationship_summary", x_column, y_column,
                observation_count=len(paired),
                pearson={"correlation": float(pearson_value), "p_value": float(pearson_p)},
                spearman={"correlation": float(spearman_value), "p_value": float(spearman_p)},
                kendall={"correlation": float(kendall_value), "p_value": float(kendall_p)},
                covariance=float(paired["x"].cov(paired["y"])),
                linear_regression={
                    "slope": float(regression.slope),
                    "intercept": float(regression.intercept),
                    "r_squared": float(regression.rvalue ** 2),
                    "r_value": float(regression.rvalue),
                    "p_value": float(regression.pvalue),
                    "standard_error": float(regression.stderr),
                },
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Relationship summary failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def build_tools(self) -> list[Any]:
        """Build LangChain numeric-numeric statistics tools."""
        @tool
        def paired_statistics(x_column: str, y_column: str) -> dict[str, Any]:
            """Calculate basic statistics for a numeric column pair."""
            return self.paired_statistics(x_column, y_column)

        @tool
        def pearson_correlation(x_column: str, y_column: str) -> dict[str, Any]:
            """Calculate Pearson linear correlation."""
            return self.pearson_correlation(x_column, y_column)

        @tool
        def spearman_correlation(x_column: str, y_column: str) -> dict[str, Any]:
            """Calculate Spearman monotonic correlation."""
            return self.spearman_correlation(x_column, y_column)

        @tool
        def kendall_correlation(x_column: str, y_column: str) -> dict[str, Any]:
            """Calculate Kendall rank correlation."""
            return self.kendall_correlation(x_column, y_column)

        @tool
        def covariance(x_column: str, y_column: str) -> dict[str, Any]:
            """Calculate sample covariance."""
            return self.covariance(x_column, y_column)

        @tool
        def linear_regression(x_column: str, y_column: str) -> dict[str, Any]:
            """Fit simple linear regression between two numeric columns."""
            return self.linear_regression(x_column, y_column)

        @tool
        def regression_residuals(x_column: str, y_column: str) -> dict[str, Any]:
            """Calculate linear regression residual diagnostics."""
            return self.regression_residuals(x_column, y_column)

        @tool
        def paired_missing_values(x_column: str, y_column: str) -> dict[str, Any]:
            """Analyze missing observations across a numeric pair."""
            return self.paired_missing_values(x_column, y_column)

        @tool
        def relationship_summary(x_column: str, y_column: str) -> dict[str, Any]:
            """Generate a complete numeric-numeric relationship summary."""
            return self.relationship_summary(x_column, y_column)

        tools = [
            paired_statistics, 
            pearson_correlation, 
            spearman_correlation,
            kendall_correlation, 
            covariance, 
            linear_regression,
            regression_residuals, 
            paired_missing_values,
            relationship_summary,
        ]
        self.logger.info(
            "Built %d numeric-numeric bivariate statistics tools.", len(tools)
        )
        return tools



# ====================================================
# Numeric-Numeric Bivariate Visualization
# ====================================================
"""
Numeric-Numeric Bivariate Visualization
=======================================
Seaborn and Matplotlib visualizations for relationships between
two numeric features.
Supported visualizations
------------------------
- Scatter plot
- Regression plot
- Hexbin plot
- 2D KDE plot
- Joint plot
- Residual plot
- Line plot
- Correlation heatmap
Design principles
-----------------
1. Pandas DataFrame remains internal to the engine.
2. The LLM supplies only column names and analytical parameters.
3. Every visualization validates both requested columns.
4. Numeric column validation is performed before plotting.
5. Matplotlib Figure objects are returned for downstream rendering.
6. Seaborn and Matplotlib are used for visualization.
7. Source DataFrame is never modified.
8. Project logging and CustomException are used consistently.
"""
from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class NumericNumericBivariateVisualization:
    """
    Create deterministic visualizations for numeric-numeric relationships.
    Parameters
    ----------
    df:
        Pandas DataFrame containing the dataset.
    logger_name:
        Logger name used by the application.
    Notes
    -----
    The engine does not modify the source DataFrame or write files.
    The DataFrame is captured internally and is never exposed to the LLM.
    """
    DEFAULT_FIGSIZE = (10, 6)
    DEFAULT_TOP_N = 10

    def __init__(
        self,
        df: pd.DataFrame,
        logger_name: str = "PandasAgentNumericNumericBIVisualization",
    ) -> None:
        """Initialize the numeric-numeric visualization engine."""
        if not isinstance(df, pd.DataFrame):
            raise CustomException("df must be a pandas DataFrame.", get_log(logger_name))
        if df.empty:
            raise CustomException("DataFrame cannot be empty.", get_log(logger_name))
        self.df = df
        self.logger = get_log(logger_name)
        self.logger.info(
            "Initialized numeric-numeric visualization engine "
            "with %d rows and %d columns.",
            df.shape[0], df.shape[1],
        )

    def _validate_column(self, column: str) -> pd.Series:
        """
        Validate that a requested column exists and is numeric.
        """
        try:
            if not isinstance(column, str) or not column.strip():
                raise ValueError("Column name must be a non-empty string.")
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' does not exist.")
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise TypeError(
                    f"Column '{column}' must be numeric. "
                    f"Detected dtype: {self.df[column].dtype}"
                )
            return self.df[column]
        except Exception as exc:
            self.logger.error(
                "Numeric column validation failed for '%s': %s", column, exc
            )
            raise CustomException(exc, self.logger) from exc

    def _validate_columns(
        self, x_column: str, y_column: str
    ) -> tuple[pd.Series, pd.Series]:
        """Validate both numeric columns."""
        try:
            if x_column == y_column:
                raise ValueError("x_column and y_column must be different.")
            return self._validate_column(x_column), self._validate_column(y_column)
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error(
                "Numeric-numeric column validation failed for '%s' and '%s': %s",
                x_column, y_column, exc,
            )
            raise CustomException(exc, self.logger) from exc

    def _get_pair_data(self, x_column: str, y_column: str) -> pd.DataFrame:
        """
        Return paired non-missing observations.
        Only the two requested columns are selected.
        """
        x, y = self._validate_columns(x_column, y_column)
        pair_df = pd.DataFrame({x_column: x, y_column: y}).dropna()
        if pair_df.empty:
            raise ValueError(
                f"No valid paired observations available for "
                f"'{x_column}' and '{y_column}'."
            )
        return pair_df

    def _create_figure(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a standard Matplotlib figure."""
        fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax

    @staticmethod
    def _format_axis(ax: plt.Axes) -> None:
        """Apply common axis formatting."""
        ax.grid(axis="both", alpha=0.25)
        ax.set_axisbelow(True)

    @track_performance
    def scatter_plot(
        self,
        x_column: str,
        y_column: str,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
        alpha: float = 0.6,
    ) -> plt.Figure:
        """
        Create a scatter plot between two numeric variables.
        Useful for:
        - Relationship inspection
        - Direction detection
        - Outlier detection
        - Non-linear pattern detection
        """
        try:
            pair_df = self._get_pair_data(x_column, y_column)
            if not 0 < alpha <= 1:
                raise ValueError("alpha must be between 0 and 1.")
            fig, ax = self._create_figure(
                title=f"Scatter Plot: {x_column} vs {y_column}",
                xlabel=x_column, ylabel=y_column, figsize=figsize,
            )
            sns.scatterplot(data=pair_df, x=x_column, y=y_column, alpha=alpha, ax=ax)
            self._format_axis(ax)
            fig.tight_layout()
            self.logger.info(
                "Created scatter plot for '%s' vs '%s'.", x_column, y_column
            )
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.exception(
                "Scatter plot failed for '%s' vs '%s'.", x_column, y_column
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def regression_plot(
        self,
        x_column: str,
        y_column: str,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
        order: int = 1,
        ci: int | None = 95,
        scatter: bool = True,
    ) -> plt.Figure:
        """
        Create a regression plot between two numeric variables.
        Parameters
        ----------
        order:
            Polynomial regression order.
        ci:
            Confidence interval percentage.
            Use None to disable confidence interval.
        scatter:
            Whether to display the underlying observations.
        """
        try:
            pair_df = self._get_pair_data(x_column, y_column)
            if order < 1:
                raise ValueError("order must be at least 1.")
            if ci is not None and not 0 <= ci <= 100:
                raise ValueError("ci must be between 0 and 100 or None.")
            fig, ax = self._create_figure(
                title=f"Regression Plot: {x_column} vs {y_column}",
                xlabel=x_column, ylabel=y_column, figsize=figsize,
            )
            sns.regplot(
                data=pair_df, x=x_column, y=y_column,
                order=order, ci=ci, scatter=scatter, ax=ax,
            )
            self._format_axis(ax)
            fig.tight_layout()
            self.logger.info(
                "Created regression plot for '%s' vs '%s'.", x_column, y_column
            )
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.exception(
                "Regression plot failed for '%s' vs '%s'.", x_column, y_column
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def hexbin_plot(
        self,
        x_column: str,
        y_column: str,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
        gridsize: int = 30,
    ) -> plt.Figure:
        """
        Create a hexbin plot for dense numeric relationships.
        Hexbin visualization is useful when a scatter plot contains
        substantial point overlap.
        """
        try:
            pair_df = self._get_pair_data(x_column, y_column)
            if gridsize < 1:
                raise ValueError("gridsize must be at least 1.")
            fig, ax = self._create_figure(
                title=f"Hexbin Plot: {x_column} vs {y_column}",
                xlabel=x_column, ylabel=y_column, figsize=figsize,
            )
            ax.hexbin(
                pair_df[x_column], pair_df[y_column],
                gridsize=gridsize, mincnt=1,
            )
            ax.set_title(
                f"Hexbin Plot: {x_column} vs {y_column}",
                fontsize=14, fontweight="bold",
            )
            self._format_axis(ax)
            fig.tight_layout()
            self.logger.info(
                "Created hexbin plot for '%s' vs '%s'.", x_column, y_column
            )
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.exception(
                "Hexbin plot failed for '%s' vs '%s'.", x_column, y_column
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def kde_2d_plot(
        self,
        x_column: str,
        y_column: str,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
        fill: bool = True,
        levels: int = 10,
    ) -> plt.Figure:
        """
        Create a two-dimensional kernel density estimate plot.
        Useful for identifying concentration regions and the shape
        of a numeric-numeric distribution.
        """
        try:
            pair_df = self._get_pair_data(x_column, y_column)
            if levels < 1:
                raise ValueError("levels must be at least 1.")
            fig, ax = self._create_figure(
                title=f"2D KDE: {x_column} vs {y_column}",
                xlabel=x_column, ylabel=y_column, figsize=figsize,
            )
            sns.kdeplot(
                data=pair_df, x=x_column, y=y_column,
                fill=fill, levels=levels, ax=ax,
            )
            self._format_axis(ax)
            fig.tight_layout()
            self.logger.info(
                "Created 2D KDE plot for '%s' vs '%s'.", x_column, y_column
            )
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.exception(
                "2D KDE plot failed for '%s' vs '%s'.", x_column, y_column
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def joint_plot(
        self,
        x_column: str,
        y_column: str,
        kind: str = "scatter",
        height: float = 7.0,
    ) -> sns.axisgrid.JointGrid:
        """
        Create a Seaborn joint plot.
        Supported kinds include:
        - scatter
        - kde
        - hist
        - hex
        - reg
        - resid
        """
        try:
            pair_df = self._get_pair_data(x_column, y_column)
            supported_kinds = {"scatter", "kde", "hist", "hex", "reg", "resid"}
            if kind not in supported_kinds:
                raise ValueError(
                    f"Unsupported joint plot kind '{kind}'. "
                    f"Supported kinds: {sorted(supported_kinds)}"
                )
            if height <= 0:
                raise ValueError("height must be greater than zero.")
            grid = sns.jointplot(
                data=pair_df, x=x_column, y=y_column, kind=kind, height=height
            )
            grid.fig.suptitle(
                f"Joint Plot: {x_column} vs {y_column}",
                y=1.02, fontsize=14, fontweight="bold",
            )
            grid.fig.tight_layout()
            self.logger.info(
                "Created joint plot for '%s' vs '%s' using kind='%s'.",
                x_column, y_column, kind,
            )
            return grid
        except CustomException:
            raise
        except Exception as exc:
            self.logger.exception(
                "Joint plot failed for '%s' vs '%s'.", x_column, y_column
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def residual_plot(
        self,
        x_column: str,
        y_column: str,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    ) -> plt.Figure:
        """
        Create a residual plot for the relationship between two
        numeric variables.
        Useful for inspecting:
        - Non-linearity
        - Heteroscedasticity
        - Systematic residual patterns
        """
        try:
            pair_df = self._get_pair_data(x_column, y_column)
            fig, ax = self._create_figure(
                title=f"Residual Plot: {x_column} vs {y_column}",
                xlabel=x_column, ylabel="Residual", figsize=figsize,
            )
            sns.residplot(
                data=pair_df, x=x_column, y=y_column,
                lowess=True, scatter_kws={"alpha": 0.6}, ax=ax,
            )
            ax.axhline(0, linestyle="--", linewidth=1)
            self._format_axis(ax)
            fig.tight_layout()
            self.logger.info(
                "Created residual plot for '%s' vs '%s'.", x_column, y_column
            )
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.exception(
                "Residual plot failed for '%s' vs '%s'.", x_column, y_column
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def line_plot(
        self,
        x_column: str,
        y_column: str,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
        sort_x: bool = True,
    ) -> plt.Figure:
        """
        Create a line plot between two numeric variables.
        Parameters
        ----------
        sort_x:
            Sort observations by the X variable before plotting.
            The source DataFrame is never modified.
        """
        try:
            pair_df = self._get_pair_data(x_column, y_column)
            if sort_x:
                pair_df = pair_df.sort_values(by=x_column)
            fig, ax = self._create_figure(
                title=f"Line Plot: {x_column} vs {y_column}",
                xlabel=x_column, ylabel=y_column, figsize=figsize,
            )
            sns.lineplot(data=pair_df, x=x_column, y=y_column, ax=ax)
            self._format_axis(ax)
            fig.tight_layout()
            self.logger.info(
                "Created line plot for '%s' vs '%s'.", x_column, y_column
            )
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.exception(
                "Line plot failed for '%s' vs '%s'.", x_column, y_column
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def correlation_heatmap(
        self,
        columns: list[str] | None = None,
        method: str = "pearson",
        figsize: tuple[int, int] = (10, 8),
        annot: bool = True,
    ) -> plt.Figure:
        """
        Create a correlation heatmap for selected numeric columns.
        Parameters
        ----------
        columns:
            Numeric columns to include.
            If None, all numeric columns are used.
        method:
            Correlation method:
            - pearson
            - spearman
            - kendall
        annot:
            Display correlation values in cells.
        """
        try:
            supported_methods = {"pearson", "spearman", "kendall"}
            if method not in supported_methods:
                raise ValueError(
                    f"Unsupported correlation method '{method}'. "
                    f"Supported methods: {sorted(supported_methods)}"
                )
            if columns is None:
                numeric_df = self.df.select_dtypes(include="number")
            else:
                if not isinstance(columns, list):
                    raise TypeError("columns must be a list of column names.")
                if not columns:
                    raise ValueError("columns cannot be empty.")
                invalid = [c for c in columns if c not in self.df.columns]
                if invalid:
                    raise ValueError(f"Columns do not exist: {invalid}")
                non_numeric = [
                    c for c in columns
                    if not pd.api.types.is_numeric_dtype(self.df[c])
                ]
                if non_numeric:
                    raise TypeError(f"Columns must be numeric: {non_numeric}")
                numeric_df = self.df[columns]
            if numeric_df.shape[1] < 2:
                raise ValueError(
                    "At least two numeric columns are required "
                    "for a correlation heatmap."
                )
            correlation = numeric_df.corr(method=method)
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(correlation, annot=annot, fmt=".2f", center=0, ax=ax)
            ax.set_title(
                f"Correlation Heatmap ({method.title()})",
                fontsize=14, fontweight="bold",
            )
            fig.tight_layout()
            self.logger.info(
                "Created %s correlation heatmap using %d columns.",
                method, numeric_df.shape[1],
            )
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.exception("Correlation heatmap failed: %s", exc)
            raise CustomException(exc, self.logger) from exc

    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools backed by this DataFrame.
        The DataFrame remains captured internally.
        The agent supplies column names and visualization parameters.
        """
        visualizer = self

        @tool
        def numeric_numeric_scatter_plot(
            x_column: str, y_column: str, alpha: float = 0.6
        ) -> Any:
            """
            Create a scatter plot between two numeric columns.
            Use for inspecting relationships, patterns,
            outliers, and non-linear behavior.
            Args:
                x_column: Numeric X-axis column.
                y_column: Numeric Y-axis column.
                alpha: Point transparency between 0 and 1.
            """
            return visualizer.scatter_plot(
                x_column=x_column, y_column=y_column, alpha=alpha
            )

        @tool
        def numeric_numeric_regression_plot(
            x_column: str, y_column: str, order: int = 1, ci: int | None = 95
        ) -> Any:
            """
            Create a regression plot between two numeric columns.
            Args:
                x_column: Numeric predictor column.
                y_column: Numeric response column.
                order: Polynomial regression order.
                ci: Confidence interval percentage or None.
            """
            return visualizer.regression_plot(
                x_column=x_column, y_column=y_column, order=order, ci=ci
            )

        @tool
        def numeric_numeric_hexbin_plot(
            x_column: str, y_column: str, gridsize: int = 30
        ) -> Any:
            """
            Create a hexbin plot for dense numeric data.
            Args:
                x_column: Numeric X-axis column.
                y_column: Numeric Y-axis column.
                gridsize: Number of hexagonal bins.
            """
            return visualizer.hexbin_plot(
                x_column=x_column, y_column=y_column, gridsize=gridsize
            )

        @tool
        def numeric_numeric_kde_plot(
            x_column: str, y_column: str, fill: bool = True, levels: int = 10
        ) -> Any:
            """
            Create a two-dimensional KDE plot.
            Useful for visualizing concentration and
            distribution density.
            Args:
                x_column: Numeric X-axis column.
                y_column: Numeric Y-axis column.
                fill: Fill KDE regions.
                levels: Number of KDE contour levels.
            """
            return visualizer.kde_2d_plot(
                x_column=x_column, y_column=y_column, fill=fill, levels=levels
            )

        @tool
        def numeric_numeric_joint_plot(
            x_column: str, y_column: str, kind: str = "scatter"
        ) -> Any:
            """
            Create a Seaborn joint plot for two numeric columns.
            Supported kinds:
            scatter, kde, hist, hex, reg, resid
            Args:
                x_column: Numeric X-axis column.
                y_column: Numeric Y-axis column.
                kind: Joint plot type.
            """
            return visualizer.joint_plot(
                x_column=x_column, y_column=y_column, kind=kind
            )

        @tool
        def numeric_numeric_residual_plot(x_column: str, y_column: str) -> Any:
            """
            Create a residual plot between two numeric columns.
            Useful for inspecting non-linearity and
            heteroscedasticity.
            Args:
                x_column: Numeric predictor column.
                y_column: Numeric response column.
            """
            return visualizer.residual_plot(x_column=x_column, y_column=y_column)

        @tool
        def numeric_numeric_line_plot(
            x_column: str, y_column: str, sort_x: bool = True
        ) -> Any:
            """
            Create a line plot between two numeric columns.
            Args:
                x_column: Numeric X-axis column.
                y_column: Numeric Y-axis column.
                sort_x: Sort X values before plotting.
            """
            return visualizer.line_plot(
                x_column=x_column, y_column=y_column, sort_x=sort_x
            )

        @tool
        def numeric_correlation_heatmap(
            columns: list[str] | None = None,
            method: str = "pearson",
            annot: bool = True,
        ) -> Any:
            """
            Create a correlation heatmap for numeric columns.
            Supported methods:
            pearson, spearman, kendall
            Args:
                columns: Numeric columns to include.
                method: Correlation method.
                annot: Display correlation values.
            """
            return visualizer.correlation_heatmap(
                columns=columns, method=method, annot=annot
            )

        tools = [
            numeric_numeric_scatter_plot,
            numeric_numeric_regression_plot,
            numeric_numeric_hexbin_plot,
            numeric_numeric_kde_plot,
            numeric_numeric_joint_plot,
            numeric_numeric_residual_plot,
            numeric_numeric_line_plot,
            numeric_correlation_heatmap,
        ]
        self.logger.info(
            "Built %d numeric-numeric visualization tools.", len(tools)
        )
        return tools


# ====================================================
# Factory
# ====================================================

def build_numeric_numeric_bivariate_statistics_tools(df: pd.DataFrame) -> list[Any]:
    """
    Create all numeric-numeric bivariate statistical LangChain tools
    for a DataFrame.
    Parameters
    ----------
    df:
        DataFrame that the tools will analyze.
    Returns
    -------
    list[Any]
        LangChain tools covering numeric-numeric bivariate analysis.
    Raises
    ------
    CustomException
        If the DataFrame is invalid.
    """
    engine = NumericNumericBivariateStatistics(df)
    engine.logger.info("Building numeric-numeric bivariate statistics tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d numeric-numeric bivariate statistics tools.", len(tools))
    return tools

def build_numeric_numeric_bivariate_visualization_tools(df: pd.DataFrame) -> list[Any]:
    """
    Build numeric-numeric bivariate visualization tools.
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
    engine = NumericNumericBivariateVisualization(df)
    engine.logger.info("Building numeric-numeric bivariate visualization tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d numeric-numeric bivariate visualization tools.", len(tools))
    return tools

# ====================================================
# Build Tools
# ====================================================

def build_numeric_numeric_bivariate_tools(df: pd.DataFrame) -> list[Any]:
    """
    Build all numeric-numeric bivariate tools.
    Includes
    --------
    - Numeric-numeric bivariate statistics
    - Numeric-numeric bivariate visualizations
    Parameters
    ----------
    df:
        Source DataFrame.
    Returns
    -------
    list[Any]
        Complete numeric-numeric bivariate tool collection.
    """
    tools: list[Any] = []
    tools.extend(build_numeric_numeric_bivariate_statistics_tools(df))
    tools.extend(build_numeric_numeric_bivariate_visualization_tools(df))
    return tools

