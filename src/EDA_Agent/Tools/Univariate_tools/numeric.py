#====================================================
# Numeric Univariate Statistics
#====================================================


"""
Numeric univariate analysis tools for the Pandas Data Agent.
This module provides deterministic analytical tools for examining a single
numeric variable at a time.
The tools are designed to be registered with LangChain/LangGraph using
``bind_tools`` and ``ToolNode``.
Supported analyses
------------------
- Basic numeric statistics
- Central tendency
- Dispersion
- Quantiles
- Distribution shape
- Missing and infinite values
- Zero values
- Negative values
- Unique values / cardinality
- IQR-based outlier analysis
- Histogram distribution
- Distribution diagnostics / normality tests
Design principles
-----------------
1. The LLM selects the tool; it does not perform the calculation.
2. Pandas/SciPy perform all deterministic calculations.
3. The DataFrame is captured by the tool factory and is never supplied
   by the LLM.
4. Every tool validates the requested column.
5. Every tool returns structured JSON-serializable data.
6. Errors are logged through the project logger.
7. Unexpected internal failures are wrapped using ``CustomException``.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy.stats import anderson, kurtosis, normaltest, skew
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class NumericUnivariateStatistics:
    """
    Deterministic numeric univariate analysis engine.
    The class owns the DataFrame and exposes methods that are subsequently
    wrapped as LangChain tools.
    Parameters
    ----------
    df:
        Pandas DataFrame on which numeric univariate analysis will be
        performed.
    logger_name:
        Name used when creating the project logger.
    Notes
    -----
    This class does not modify the DataFrame and does not write Excel
    workbooks. It is intended for dynamic analytical queries from the
    Pandas Data Agent.
    """
    HISTOGRAM_BINS = 20
    LOW_CARDINALITY_THRESHOLD = 10
    QUANTILE_LEVELS = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]

    def __init__(self, df: pd.DataFrame, logger_name: str = "PandasAgentNumericUNI") -> None:
        """
        Initialize the numeric univariate analysis engine.
        Parameters
        ----------
        df:
            DataFrame containing the dataset to analyze.
        logger_name:
            Logger name used by the application logging infrastructure.
        Raises
        ------
        CustomException
            If the DataFrame is missing or empty.
        """
        if df is None:
            raise CustomException("DataFrame cannot be None.", get_log(logger_name))
        if df.empty:
            raise CustomException("DataFrame cannot be empty.", get_log(logger_name))
        self.df = df
        self.logger = get_log(logger_name)

    def _validate_column(self, column: str) -> pd.Series:
        """
        Validate that a requested column exists and can be treated as numeric.
        Parameters
        ----------
        column:
            Name of the DataFrame column.
        Returns
        -------
        pd.Series
            Original DataFrame column.
        Raises
        ------
        CustomException
            If the column does not exist or contains no usable numeric data.
        """
        try:
            if not column:
                raise ValueError("Column name cannot be empty.")
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
            series = pd.to_numeric(self.df[column], errors="coerce")
            valid = series.replace([np.inf, -np.inf], np.nan).dropna()
            if valid.empty:
                raise ValueError(f"Column '{column}' contains no valid numeric values.")
            return self.df[column]
        except Exception as exc:
            self.logger.error("Numeric column validation failed for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    def _get_numeric_series(self, column: str) -> pd.Series:
        """
        Convert a DataFrame column to clean numeric observations.
        Invalid numeric values, NaN values, positive infinity, and negative
        infinity are excluded from the returned series.
        Parameters
        ----------
        column:
            Name of the numeric column.
        Returns
        -------
        pd.Series
            Clean numeric observations.
        Raises
        ------
        CustomException
            If the requested column is invalid.
        """
        self._validate_column(column)
        try:
            series = pd.to_numeric(self.df[column], errors="coerce")
            return series.replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as exc:
            self.logger.error("Failed to create numeric series for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @staticmethod
    def _safe_percentage(numerator: int | float, denominator: int | float) -> float:
        """
        Calculate a percentage safely.
        Parameters
        ----------
        numerator:
            Numerator of the percentage calculation.
        denominator:
            Denominator of the percentage calculation.
        Returns
        -------
        float
            Percentage rounded to four decimal places, or ``0.0`` when the
            denominator is zero.
        """
        if denominator == 0:
            return 0.0
        return round(float(numerator) / float(denominator) * 100, 4)

    @staticmethod
    @track_performance
    def _json_safe(value: Any) -> Any:
        """
        Convert common NumPy/Pandas scalar values into JSON-safe values.
        Parameters
        ----------
        value:
            Value that may contain a NumPy scalar or NaN.
        Returns
        -------
        Any
            JSON-compatible representation.
        """
        if value is None:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @track_performance
    def _result(self, tool_name: str, column: str, **values: Any) -> dict[str, Any]:
        """
        Build a standardized successful tool response.
        Parameters
        ----------
        tool_name:
            Name of the analytical tool.
        column:
            Column analyzed.
        **values:
            Tool-specific analytical results.
        Returns
        -------
        dict[str, Any]
            JSON-compatible tool result.
        """
        result: dict[str, Any] = {
            "status": "success",
            "tool": tool_name,
            "analysis_type": "univariate",
            "data_type": "numeric",
            "column": column,
        }
        for key, value in values.items():
            if isinstance(value, dict):
                result[key] = {k: self._json_safe(v) for k, v in value.items()}
            elif isinstance(value, list):
                result[key] = [
                    {k: self._json_safe(v) for k, v in item.items()} if isinstance(item, dict) else self._json_safe(item)
                    for item in value
                ]
            else:
                result[key] = self._json_safe(value)
        return result

    @track_performance
    def numeric_statistics(self, column: str) -> dict[str, Any]:
        """
        Calculate basic statistics and data-quality information for one
        numeric column.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Includes row count, valid count, missing count, missing
            percentage, unique count, unique percentage, dtype, minimum,
            maximum, mean, and median.
        Notes
        -----
        Infinity values are treated separately from missing values.
        """
        try:
            self._validate_column(column)
            raw = pd.to_numeric(self.df[column], errors="coerce")
            infinite_count = int(np.isinf(raw).sum())
            valid = raw.replace([np.inf, -np.inf], np.nan).dropna()
            total_rows, valid_count = len(raw), len(valid)
            missing_count = int(raw.isna().sum())
            unique_count = int(valid.nunique())
            self.logger.info("Computed numeric statistics for column '%s'.", column)
            return self._result(
                "numeric_statistics", column,
                dtype=str(self.df[column].dtype),
                total_rows=total_rows,
                valid_count=valid_count,
                missing_count=missing_count,
                missing_percentage=self._safe_percentage(missing_count, total_rows),
                infinite_count=infinite_count,
                unique_count=unique_count,
                unique_percentage=self._safe_percentage(unique_count, valid_count),
                minimum=valid.min() if not valid.empty else None,
                maximum=valid.max() if not valid.empty else None,
                mean=valid.mean() if not valid.empty else None,
                median=valid.median() if not valid.empty else None,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed numeric statistics for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def central_tendency(self, column: str) -> dict[str, Any]:
        """
        Calculate mean, median, and mode for a numeric column.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Mean, median, mode, mode count, and mode percentage.
        Notes
        -----
        When multiple modes exist, the first mode returned by Pandas is
        reported. The complete multimodal result can be added later if
        required by the response schema.
        """
        try:
            series = self._get_numeric_series(column)
            mode_values = series.mode()
            if mode_values.empty:
                mode, mode_count = None, 0
            else:
                mode = mode_values.iloc[0]
                mode_count = int((series == mode).sum())
            self.logger.info("Computed central tendency for column '%s'.", column)
            return self._result(
                "central_tendency", column,
                mean=series.mean(),
                median=series.median(),
                mode=mode,
                mode_count=mode_count,
                mode_percentage=self._safe_percentage(mode_count, len(series)),
                mode_count_total=int(len(mode_values)),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed central tendency for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def dispersion_statistics(self, column: str) -> dict[str, Any]:
        """
        Calculate dispersion statistics for a numeric column.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Variance, standard deviation, minimum, maximum, range, IQR,
            mean absolute deviation, and coefficient of variation.
        Notes
        -----
        Pandas sample variance and sample standard deviation are used by
        default. Coefficient of variation is returned as ``None`` when the
        mean is effectively zero.
        """
        try:
            series = self._get_numeric_series(column)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            mean, std = series.mean(), series.std()
            cv = std / abs(mean) if not np.isclose(mean, 0) else None
            self.logger.info("Computed dispersion statistics for column '%s'.", column)
            return self._result(
                "dispersion_statistics", column,
                variance=series.var(),
                standard_deviation=std,
                minimum=series.min(),
                maximum=series.max(),
                range=series.max() - series.min(),
                iqr=q3 - q1,
                mean_absolute_deviation=np.mean(np.abs(series - mean)),
                coefficient_of_variation=cv,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed dispersion statistics for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def quantiles(self, column: str) -> dict[str, Any]:
        """
        Calculate important quantiles for a numeric column.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Quantiles from 0th percentile through 100th percentile,
            including commonly useful tail and quartile levels.
        """
        try:
            series = self._get_numeric_series(column)
            values = series.quantile(self.QUANTILE_LEVELS)
            quantile_result = {
                "q0": values.loc[0.00], "q01": values.loc[0.01], "q05": values.loc[0.05],
                "q10": values.loc[0.10], "q25": values.loc[0.25], "q50": values.loc[0.50],
                "q75": values.loc[0.75], "q90": values.loc[0.90], "q95": values.loc[0.95],
                "q99": values.loc[0.99], "q100": values.loc[1.00],
            }
            self.logger.info("Computed quantiles for column '%s'.", column)
            return self._result("quantiles", column, quantiles=quantile_result)
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed quantile calculation for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def distribution_shape(self, column: str) -> dict[str, Any]:
        """
        Calculate skewness, kurtosis, and mean-median relationships.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Skewness, Fisher kurtosis, mean, median, mean-median difference,
            and mean-median ratio.
        Raises
        ------
        CustomException
            If fewer than three valid observations are available.
        """
        try:
            series = self._get_numeric_series(column)
            if len(series) < 3:
                raise ValueError(
                    f"At least 3 valid observations are required for "
                    f"distribution-shape analysis of '{column}'."
                )
            mean, median = series.mean(), series.median()
            result_ratio = mean / median if not np.isclose(median, 0) else None
            self.logger.info("Computed distribution shape for column '%s'.", column)
            return self._result(
                "distribution_shape", column,
                sample_size=len(series),
                skewness=skew(series, bias=False),
                kurtosis=kurtosis(series, fisher=True, bias=False),
                mean=mean,
                median=median,
                mean_median_difference=mean - median,
                mean_median_ratio=result_ratio,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed distribution shape for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def missing_values(self, column: str) -> dict[str, Any]:
        """
        Analyze missing and infinite values in a numeric column.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Total rows, missing count, missing percentage, positive/negative
            infinity counts, total infinity count, and valid count.
        Notes
        -----
        Missing values and infinite values are reported separately because
        they represent different data-quality conditions.
        """
        try:
            self._validate_column(column)
            raw = pd.to_numeric(self.df[column], errors="coerce")
            pos_inf = int(np.isposinf(raw).sum())
            neg_inf = int(np.isneginf(raw).sum())
            infinite_count = pos_inf + neg_inf
            missing_count = int(raw.isna().sum())
            total_rows = len(raw)
            valid_count = total_rows - missing_count - infinite_count
            self.logger.info("Computed missing-value metrics for '%s'.", column)
            return self._result(
                "missing_values", column,
                total_rows=total_rows,
                missing_count=missing_count,
                missing_percentage=self._safe_percentage(missing_count, total_rows),
                positive_infinity_count=pos_inf,
                negative_infinity_count=neg_inf,
                infinite_count=infinite_count,
                valid_count=valid_count,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed missing-value analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def zero_values(self, column: str) -> dict[str, Any]:
        """
        Calculate the count and percentage of zero values.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Valid observations, zero count, and zero percentage.
        """
        try:
            series = self._get_numeric_series(column)
            zero_count = int((series == 0).sum())
            self.logger.info("Computed zero-value metrics for '%s'.", column)
            return self._result(
                "zero_values", column,
                valid_count=len(series),
                zero_count=zero_count,
                zero_percentage=self._safe_percentage(zero_count, len(series)),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed zero-value analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def negative_values(self, column: str) -> dict[str, Any]:
        """
        Calculate the count and percentage of negative values.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Valid observations, negative count, negative percentage, and
            minimum value.
        """
        try:
            series = self._get_numeric_series(column)
            negative_count = int((series < 0).sum())
            self.logger.info("Computed negative-value metrics for '%s'.", column)
            return self._result(
                "negative_values", column,
                valid_count=len(series),
                negative_count=negative_count,
                negative_percentage=self._safe_percentage(negative_count, len(series)),
                minimum_value=series.min(),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed negative-value analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def unique_values(self, column: str) -> dict[str, Any]:
        """
        Analyze cardinality and uniqueness of a numeric column.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Valid count, unique count, unique percentage, duplicate count,
            constant-column indicator, and low-cardinality indicator.
        """
        try:
            series = self._get_numeric_series(column)
            valid_count = len(series)
            unique_count = int(series.nunique())
            self.logger.info("Computed unique-value metrics for '%s'.", column)
            return self._result(
                "unique_values", column,
                valid_count=valid_count,
                unique_count=unique_count,
                unique_percentage=self._safe_percentage(unique_count, valid_count),
                duplicate_count=valid_count - unique_count,
                constant=unique_count <= 1,
                low_cardinality=unique_count <= self.LOW_CARDINALITY_THRESHOLD,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed unique-value analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def outlier_analysis(self, column: str) -> dict[str, Any]:
        """
        Detect IQR-based outliers in a numeric column.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Q1, Q3, IQR, lower and upper bounds, total outliers, lower
            outliers, upper outliers, and their percentages.
        Notes
        -----
        The standard Tukey rule is used:
        ``lower_bound = Q1 - 1.5 * IQR``
        ``upper_bound = Q3 + 1.5 * IQR``
        """
        try:
            series = self._get_numeric_series(column)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            lower_out = int((series < lower_bound).sum())
            upper_out = int((series > upper_bound).sum())
            outlier_count = lower_out + upper_out
            n = len(series)
            self.logger.info("Computed IQR outliers for '%s'.", column)
            return self._result(
                "outlier_analysis", column,
                valid_count=n,
                q1=q1, q3=q3, iqr=iqr,
                lower_bound=lower_bound, upper_bound=upper_bound,
                outlier_count=outlier_count,
                outlier_percentage=self._safe_percentage(outlier_count, n),
                lower_outlier_count=lower_out,
                upper_outlier_count=upper_out,
                lower_outlier_percentage=self._safe_percentage(lower_out, n),
                upper_outlier_percentage=self._safe_percentage(upper_out, n),
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed outlier analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def histogram_distribution(self, column: str, bins: int = HISTOGRAM_BINS) -> dict[str, Any]:
        """
        Calculate a histogram distribution for a numeric column.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        bins:
            Number of equal-width histogram bins.
        Returns
        -------
        dict[str, Any]
            Histogram bin boundaries, midpoint, count, and percentage.
        Notes
        -----
        This tool returns histogram data rather than creating a matplotlib
        figure. The response/visualization layer can use this structured
        data to render a chart.
        """
        try:
            series = self._get_numeric_series(column)
            if bins < 1:
                raise ValueError("Number of bins must be at least 1.")
            if len(series) < 2:
                raise ValueError(
                    f"At least two valid observations are required "
                    f"for histogram analysis of '{column}'."
                )
            if series.nunique() == 1:
                value = series.iloc[0]
                histogram = [{
                    "bin_start": value, "bin_end": value, "bin_midpoint": value,
                    "count": len(series), "percentage": 100.0,
                }]
            else:
                counts, edges = np.histogram(series.to_numpy(), bins=bins)
                total = int(counts.sum())
                histogram = [
                    {
                        "bin_start": edges[i],
                        "bin_end": edges[i + 1],
                        "bin_midpoint": (edges[i] + edges[i + 1]) / 2,
                        "count": int(c),
                        "percentage": self._safe_percentage(c, total),
                    }
                    for i, c in enumerate(counts)
                ]
            self.logger.info("Computed histogram distribution for '%s'.", column)
            return self._result(
                "histogram_distribution", column,
                bins=bins, valid_count=len(series), distribution=histogram,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed histogram analysis for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def distribution_diagnostics(self, column: str) -> dict[str, Any]:
        """
        Perform statistical diagnostics for normality.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        dict[str, Any]
            Sample size, D'Agostino-Pearson normality test statistic and
            p-value where applicable, plus the Anderson-Darling statistic.
        Notes
        -----
        The D'Agostino-Pearson test requires at least eight observations.
        The Anderson-Darling statistic is reported without independently
        interpreting the critical-value threshold.
        The response layer should use the returned statistical evidence
        rather than recomputing the test.
        """
        try:
            series = self._get_numeric_series(column)
            sample_size = len(series)
            if sample_size < 3:
                raise ValueError(
                    f"At least 3 valid observations are required for "
                    f"distribution diagnostics of '{column}'."
                )
            normaltest_statistic = normaltest_p_value = anderson_statistic = normaltest_name = None
            if sample_size >= 8 and series.nunique() > 1:
                try:
                    result = normaltest(series)
                    normaltest_statistic, normaltest_p_value = result.statistic, result.pvalue
                    normaltest_name = "dagostino_pearson"
                except Exception as exc:
                    self.logger.warning("D'Agostino-Pearson test failed for '%s': %s", column, exc)
            if series.nunique() > 1:
                try:
                    anderson_statistic = anderson(series, dist="norm").statistic
                except Exception as exc:
                    self.logger.warning("Anderson-Darling test failed for '%s': %s", column, exc)
            self.logger.info("Computed distribution diagnostics for '%s'.", column)
            return self._result(
                "distribution_diagnostics", column,
                sample_size=sample_size,
                normaltest=normaltest_name,
                normaltest_statistic=normaltest_statistic,
                normaltest_p_value=normaltest_p_value,
                anderson_darling_statistic=anderson_statistic,
            )
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed distribution diagnostics for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools backed by this DataFrame.
        Returns
        -------
        list[Any]
            List of LangChain-compatible tools for numeric univariate
            analysis.
        Notes
        -----
        The returned functions capture this engine instance. Therefore,
        the LLM only needs to provide analytical parameters such as the
        column name; it never receives the DataFrame itself.
        """
        @tool
        def numeric_statistics(column: str) -> dict[str, Any]:
            """
            Calculate basic statistics for one numeric column.
            Use this for questions about count, missing values, unique
            values, minimum, maximum, mean, or median.
            """
            return self.numeric_statistics(column)

        @tool
        def central_tendency(column: str) -> dict[str, Any]:
            """
            Calculate mean, median, and mode for one numeric column.
            Use this when the user asks about the typical or central value
            of a numeric variable.
            """
            return self.central_tendency(column)

        @tool
        def dispersion_statistics(column: str) -> dict[str, Any]:
            """
            Calculate variance, standard deviation, range, IQR, MAD, and
            coefficient of variation for one numeric column.
            """
            return self.dispersion_statistics(column)

        @tool
        def quantiles(column: str) -> dict[str, Any]:
            """
            Calculate percentile and quartile values for one numeric column.
            Includes Q1, median, Q3, tail percentiles, minimum, and maximum.
            """
            return self.quantiles(column)

        @tool
        def distribution_shape(column: str) -> dict[str, Any]:
            """
            Analyze skewness, kurtosis, and mean-versus-median behavior of
            one numeric column.
            """
            return self.distribution_shape(column)

        @tool
        def missing_values(column: str) -> dict[str, Any]:
            """
            Analyze missing and infinite values in one numeric column.
            """
            return self.missing_values(column)

        @tool
        def zero_values(column: str) -> dict[str, Any]:
            """
            Count zero values and calculate their percentage in one numeric
            column.
            """
            return self.zero_values(column)

        @tool
        def negative_values(column: str) -> dict[str, Any]:
            """
            Count negative values and calculate their percentage in one
            numeric column.
            """
            return self.negative_values(column)

        @tool
        def unique_values(column: str) -> dict[str, Any]:
            """
            Analyze unique values, duplicates, cardinality, constant status,
            and low-cardinality status for one numeric column.
            """
            return self.unique_values(column)

        @tool
        def outlier_analysis(column: str) -> dict[str, Any]:
            """
            Detect IQR-based lower and upper outliers in one numeric column.
            """
            return self.outlier_analysis(column)

        @tool
        def histogram_distribution(column: str, bins: int = self.HISTOGRAM_BINS) -> dict[str, Any]:
            """
            Calculate histogram bin counts and percentages for one numeric
            column. Returns structured data suitable for visualization.
            """
            return self.histogram_distribution(column, bins)

        @tool
        def distribution_diagnostics(column: str) -> dict[str, Any]:
            """
            Perform D'Agostino-Pearson and Anderson-Darling distribution
            diagnostics for one numeric column.
            """
            return self.distribution_diagnostics(column)

        tools = [
            numeric_statistics, 
            central_tendency, 
            dispersion_statistics, 
            quantiles,
            distribution_shape, 
            missing_values, 
            zero_values, 
            negative_values,
            unique_values, 
            outlier_analysis, 
            histogram_distribution, 
            distribution_diagnostics,
        ]

        return tools

#======================================================================================================================

#====================================================
# Numeric Univariate Visualization
#====================================================


"""
Numeric univariate visualization tools.
Provides deterministic numeric visualization operations using
Pandas, Seaborn, and Matplotlib.
The visualization methods are designed to be exposed to an LLM
through LangChain ``@tool`` wrappers while keeping the DataFrame
internal to the tool engine.
Supported visualizations
------------------------
- Histogram
- Boxplot
- Violin plot
- KDE plot
- Normal Q-Q plot
- ECDF plot
The tools validate numeric columns before visualization and use the
project logging and exception-handling conventions.
"""
from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from scipy import stats
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class NumericUnivariateVisualization:
    """
    Deterministic numeric univariate visualization engine.
    Parameters
    ----------
    df:
        Pandas DataFrame containing the dataset.
    Examples
    --------
    >>> engine = NumericUnivariateVisualization(df)
    >>> fig = engine.histogram("income")
    >>> fig = engine.boxplot("age")
    """
    DEFAULT_FIGSIZE = (10, 6)
    DEFAULT_BINS = 20
    DEFAULT_KDE_POINTS = 200

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the numeric visualization engine.
        Parameters
        ----------
        df:
            DataFrame containing the source dataset.
        Raises
        ------
        ValueError
            If the DataFrame is None or empty.
        """
        if df is None:
            raise ValueError("DataFrame cannot be None.")
        if df.empty:
            raise ValueError("DataFrame cannot be empty.")
        self.df = df.copy()
        self.logger = get_log("NumericUnivariateVisualization")
        self.logger.info(
            "Initialized numeric visualization engine with %d rows and %d columns.",
            self.df.shape[0], self.df.shape[1],
        )

    def _validate_column(self, column: str) -> None:
        """
        Validate that a requested column exists.
        Parameters
        ----------
        column:
            Column name to validate.
        Raises
        ------
        ValueError
            If the column does not exist.
        """
        if not column:
            raise ValueError("Column name cannot be empty.")
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    def _get_numeric_series(self, column: str) -> pd.Series:
        """
        Validate and return a cleaned numeric series.
        Parameters
        ----------
        column:
            Numeric column name.
        Returns
        -------
        pandas.Series
            Numeric series with missing and infinite values removed.
        Raises
        ------
        ValueError
            If the column is not numeric or contains no valid values.
        """
        self._validate_column(column)
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' must be numeric.")
        series = (
            pd.to_numeric(self.df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if series.empty:
            raise ValueError(f"Column '{column}' contains no valid numeric values.")
        return series

    def _create_figure(self, title: str) -> tuple[plt.Figure, plt.Axes]:
        """
        Create a standardized Matplotlib figure.
        Parameters
        ----------
        title:
            Figure title.
        Returns
        -------
        tuple
            Matplotlib Figure and Axes objects.
        """
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        ax.set_title(title)
        ax.grid(visible=True, alpha=0.25)
        return fig, ax

    def _finalize_figure(self, fig: plt.Figure) -> plt.Figure:
        """
        Apply final layout adjustments to a figure.
        Parameters
        ----------
        fig:
            Matplotlib figure.
        Returns
        -------
        matplotlib.figure.Figure
            Finalized figure.
        """
        fig.tight_layout()
        return fig

    @track_performance
    def histogram(self, column: str, bins: int = DEFAULT_BINS, kde: bool = False) -> plt.Figure:
        """
        Generate a histogram for a numeric column.
        Parameters
        ----------
        column:
            Numeric column to visualize.
        bins:
            Number of histogram bins.
        kde:
            Whether to overlay a kernel density estimate.
        Returns
        -------
        matplotlib.figure.Figure
            Generated histogram figure.
        Raises
        ------
        ValueError
            If the column is invalid or bins are invalid.
        """
        try:
            series = self._get_numeric_series(column)
            if bins < 1:
                raise ValueError("bins must be greater than zero.")
            fig, ax = self._create_figure(f"Distribution of {column}")
            sns.histplot(series, bins=bins, kde=kde, ax=ax)
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            self._finalize_figure(fig)
            self.logger.info("Generated histogram for column '%s'.", column)
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed to generate histogram for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def boxplot(self, column: str, showfliers: bool = True) -> plt.Figure:
        """
        Generate a boxplot for a numeric column.
        Parameters
        ----------
        column:
            Numeric column to visualize.
        showfliers:
            Whether to display observations classified as outliers.
        Returns
        -------
        matplotlib.figure.Figure
            Generated boxplot figure.
        Notes
        -----
        Seaborn uses Tukey-style boxplot statistics by default.
        """
        try:
            series = self._get_numeric_series(column)
            fig, ax = self._create_figure(f"Boxplot of {column}")
            sns.boxplot(x=series, showfliers=showfliers, ax=ax)
            ax.set_xlabel(column)
            self._finalize_figure(fig)
            self.logger.info("Generated boxplot for column '%s'.", column)
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed to generate boxplot for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def violinplot(self, column: str, inner: str = "box") -> plt.Figure:
        """
        Generate a violin plot for a numeric column.
        Parameters
        ----------
        column:
            Numeric column to visualize.
        inner:
            Internal representation displayed inside the violin.
            Common values include:
            - ``"box"``
            - ``"quartile"``
            - ``"point"``
            - ``"stick"``
            - ``None``
        Returns
        -------
        matplotlib.figure.Figure
            Generated violin plot.
        Raises
        ------
        ValueError
            If the column contains fewer than two unique values.
        """
        try:
            series = self._get_numeric_series(column)
            if series.nunique() < 2:
                raise ValueError(f"Violin plot requires at least two unique values in '{column}'.")
            fig, ax = self._create_figure(f"Distribution of {column}")
            sns.violinplot(x=series, inner=inner, ax=ax)
            ax.set_xlabel(column)
            self._finalize_figure(fig)
            self.logger.info("Generated violin plot for column '%s'.", column)
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed to generate violin plot for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def kde_plot(self, column: str, fill: bool = True) -> plt.Figure:
        """
        Generate a kernel density estimation plot.
        Parameters
        ----------
        column:
            Numeric column to visualize.
        fill:
            Whether to fill the area under the KDE curve.
        Returns
        -------
        matplotlib.figure.Figure
            Generated KDE figure.
        Raises
        ------
        ValueError
            If the column has fewer than two unique values.
        """
        try:
            series = self._get_numeric_series(column)
            if series.nunique() < 2:
                raise ValueError(f"KDE requires at least two unique values in '{column}'.")
            fig, ax = self._create_figure(f"Kernel Density Estimate of {column}")
            sns.kdeplot(x=series, fill=fill, ax=ax)
            ax.set_xlabel(column)
            ax.set_ylabel("Density")
            self._finalize_figure(fig)
            self.logger.info("Generated KDE plot for column '%s'.", column)
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed to generate KDE plot for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def qq_plot(self, column: str) -> plt.Figure:
        """
        Generate a normal Q-Q plot.
        Parameters
        ----------
        column:
            Numeric column to analyze.
        Returns
        -------
        matplotlib.figure.Figure
            Generated normal Q-Q plot.
        Notes
        -----
        The plot compares observed quantiles with theoretical normal
        distribution quantiles.
        """
        try:
            series = self._get_numeric_series(column)
            if len(series) < 3:
                raise ValueError(f"At least three observations are required for QQ plot of '{column}'.")
            if series.nunique() < 2:
                raise ValueError(f"QQ plot requires variation in '{column}'.")
            fig, ax = self._create_figure(f"Normal Q-Q Plot of {column}")
            stats.probplot(series, dist="norm", plot=ax)
            ax.set_title(f"Normal Q-Q Plot of {column}")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Ordered Values")
            self._finalize_figure(fig)
            self.logger.info("Generated QQ plot for column '%s'.", column)
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed to generate QQ plot for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def ecdf_plot(self, column: str) -> plt.Figure:
        """
        Generate an empirical cumulative distribution function plot.
        Parameters
        ----------
        column:
            Numeric column to visualize.
        Returns
        -------
        matplotlib.figure.Figure
            Generated ECDF figure.
        """
        try:
            series = self._get_numeric_series(column)
            fig, ax = self._create_figure(f"Empirical Cumulative Distribution of {column}")
            sns.ecdfplot(x=series, ax=ax)
            ax.set_xlabel(column)
            ax.set_ylabel("Cumulative Probability")
            self._finalize_figure(fig)
            self.logger.info("Generated ECDF plot for column '%s'.", column)
            return fig
        except CustomException:
            raise
        except Exception as exc:
            self.logger.error("Failed to generate ECDF plot for '%s': %s", column, exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools for numeric visualizations.
        Returns
        -------
        list[Any]
            List of LangChain structured tools.
        """
        @tool
        def histogram(column: str, bins: int = self.DEFAULT_BINS, kde: bool = False) -> plt.Figure:
            """
            Generate a Seaborn histogram for a numeric column.
            Use this when the user asks to visualize the distribution
            or frequency of a numeric variable.
            """
            return self.histogram(column=column, bins=bins, kde=kde)

        @tool
        def boxplot(column: str, showfliers: bool = True) -> plt.Figure:
            """
            Generate a Seaborn boxplot for a numeric column.
            Use this when the user asks about spread, quartiles,
            or potential outliers.
            """
            return self.boxplot(column=column, showfliers=showfliers)

        @tool
        def violinplot(column: str, inner: str = "box") -> plt.Figure:
            """
            Generate a Seaborn violin plot for a numeric column.
            Use this when the user wants to inspect the shape and
            density of a numeric distribution.
            """
            return self.violinplot(column=column, inner=inner)

        @tool
        def kde_plot(column: str, fill: bool = True) -> plt.Figure:
            """
            Generate a Seaborn kernel density plot for a numeric column.
            Use this when the user asks to visualize probability density
            or the smooth shape of a numeric distribution.
            """
            return self.kde_plot(column=column, fill=fill)

        @tool
        def qq_plot(column: str) -> plt.Figure:
            """
            Generate a normal Q-Q plot for a numeric column.
            Use this when the user wants to visually assess whether a
            numeric distribution approximately follows a normal
            distribution.
            """
            return self.qq_plot(column=column)

        @tool
        def ecdf_plot(column: str) -> plt.Figure:
            """
            Generate an empirical cumulative distribution plot.
            Use this when the user wants to understand cumulative
            distribution behavior or percentile relationships.
            """
            return self.ecdf_plot(column=column)

        tools = [
                 histogram, 
                 boxplot, 
                 violinplot, 
                 kde_plot, 
                 qq_plot, 
                 ecdf_plot,
                 ]
        
        self.logger.info("Built %d numeric visualization tools.", len(tools))
        return tools

#====================================================================================================

#====================================================
# Factory
#====================================================
    

def build_numeric_univariate_statistics_tools(df: pd.DataFrame) -> list[Any]:
    """
    Create all numeric univariate LangChain tools for a DataFrame.
    Parameters
    ----------
    df:
        DataFrame that the tools will analyze.
    Returns
    -------
    list[Any]
        LangChain tools covering numeric univariate analysis.
    Raises
    ------
    CustomException
        If the DataFrame is invalid.
    """
    engine = NumericUnivariateStatistics(df)
    engine.logger.info("Building numeric univariate statistics tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d numeric univariate statistics tools.", len(tools))
    return tools


def build_numeric_univariate_visualization_tools(df: pd.DataFrame) -> list[Any]:
    """
    Build numeric univariate visualization tools.
    Parameters
    ----------
    df:
        Source DataFrame.
    Returns
    -------
    list[Any]
        LangChain visualization tools.
    """
    engine = NumericUnivariateVisualization(df)
    engine.logger.info("Building numeric univariate visualization tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d numeric univariate visualization tools.", len(tools))
    return tools


#====================================================
# Build Tools
#====================================================

def build_univariate_tools(df: pd.DataFrame) -> list[Any]:
    """
    Build all univariate tools.
    """
    tools: list[Any] = []
    tools.extend(build_numeric_univariate_statistics_tools(df))
    tools.extend(build_numeric_univariate_visualization_tools(df))
    return tools
    